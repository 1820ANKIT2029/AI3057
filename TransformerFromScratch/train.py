import tensorflow as tf
from tqdm import tqdm
import os
import warnings

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from model import build_transformer
from config import get_config

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def causal_mask_tf(size):
    """
    Creates a causal mask for the decoder.
    This prevents the decoder from attending to future tokens.
    The mask is a lower triangular matrix.

    Args:
        size (int): The sequence length.

    Returns:
        tf.Tensor: A boolean tensor of shape (1, size, size).
    """
    # Create a lower triangular matrix where allowed positions are True.
    mask = tf.linalg.band_part(tf.ones((1, size, size), dtype=tf.bool), -1, 0)
    return mask

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    """
    Loads and preprocesses the dataset using TensorFlow's tf.data pipeline.

    Args:
        config (dict): A configuration dictionary with parameters like datasource,
                       languages, sequence length, batch size, etc.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (tf.data.Dataset): The batched training dataset.
            - val_dataloader (tf.data.Dataset): The batched validation dataset.
            - tokenizer_src (Tokenizer): The source language tokenizer.
            - tokenizer_tgt (Tokenizer): The target language tokenizer.
    """
    # Load the raw dataset from Hugging Face
    ds_raw = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    # Build or load tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Manually extract source and target texts and create a tf.data.Dataset
    src_texts = [item['translation'][config['lang_src']] for item in ds_raw]
    tgt_texts = [item['translation'][config['lang_tgt']] for item in ds_raw]

    ds_tf = tf.data.Dataset.from_tensor_slices({
        'src_text': src_texts,
        'tgt_text': tgt_texts
    })

    # Shuffle the dataset and split into training and validation sets
    ds_tf = ds_tf.shuffle(buffer_size=len(ds_raw), reshuffle_each_iteration=False)
    train_size = int(0.9 * len(ds_raw))
    train_ds = ds_tf.take(train_size)
    val_ds = ds_tf.skip(train_size)

    # --- Start of the tf.data preprocessing pipeline ---

    seq_len = config['seq_len']
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']

    # 1. Tokenization Step: Map a function to tokenize text pairs.
    # We use tf.py_function because the tokenizer's .encode method is not a native TF op.
    def tokenize_pairs(example):
        def _py_tokenize(src_text_tensor, tgt_text_tensor):
            src_text = src_text_tensor.numpy().decode('utf-8')
            tgt_text = tgt_text_tensor.numpy().decode('utf-8')
            src_ids = tokenizer_src.encode(src_text).ids
            tgt_ids = tokenizer_tgt.encode(tgt_text).ids
            return tf.constant(src_ids, dtype=tf.int64), tf.constant(tgt_ids, dtype=tf.int64)

        src_ids, tgt_ids = tf.py_function(
            _py_tokenize,
            [example['src_text'], example['tgt_text']],
            [tf.int64, tf.int64]
        )
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_text': example['src_text'],
            'tgt_text': example['tgt_text']
        }

    train_ds = train_ds.map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)

    # 2. Filter Step: Remove sequences that are too long.
    def filter_long_sequences(x):
        return tf.logical_and(
            tf.size(x['src_ids']) <= seq_len - 2, # For [SOS] and [EOS]
            tf.size(x['tgt_ids']) <= seq_len - 1  # For [SOS]
        )

    train_ds = train_ds.filter(filter_long_sequences)
    val_ds = val_ds.filter(filter_long_sequences)

    # 3. Formatting Step: Add special tokens, pad, and create masks.
    # This step uses pure TensorFlow operations for better performance.
    sos_token_id = tf.constant([tokenizer_tgt.token_to_id("[SOS]")], dtype=tf.int64)
    eos_token_id = tf.constant([tokenizer_tgt.token_to_id("[EOS]")], dtype=tf.int64)
    pad_token_id = tf.constant(tokenizer_tgt.token_to_id("[PAD]"), dtype=tf.int64)

    def format_dataset(x):
        # --- Encoder Input ---
        enc_padding_size = seq_len - tf.size(x['src_ids']) - 2
        encoder_input = tf.concat([
            sos_token_id,
            x['src_ids'],
            eos_token_id,
            tf.fill([enc_padding_size], pad_token_id)
        ], axis=0)

        # --- Decoder Input ---
        dec_padding_size = seq_len - tf.size(x['tgt_ids']) - 1
        decoder_input = tf.concat([
            sos_token_id,
            x['tgt_ids'],
            tf.fill([dec_padding_size], pad_token_id)
        ], axis=0)

        # --- Label (what the model is trained to predict) ---
        label_padding_size = seq_len - tf.size(x['tgt_ids']) - 1
        label = tf.concat([
            x['tgt_ids'],
            eos_token_id,
            tf.fill([label_padding_size], pad_token_id)
        ], axis=0)

        # Explicitly set the shape of the tensors.
        encoder_input.set_shape([seq_len])
        decoder_input.set_shape([seq_len])
        label.set_shape([seq_len])

        # --- Create Masks ---
        encoder_mask = tf.cast(tf.not_equal(encoder_input, pad_token_id), tf.int64)
        encoder_mask = tf.expand_dims(tf.expand_dims(encoder_mask, 0), 0)  # Shape: (1, 1, seq_len)

        decoder_padding_mask = tf.cast(tf.not_equal(decoder_input, pad_token_id), tf.bool)
        decoder_padding_mask = tf.expand_dims(decoder_padding_mask, 0) # Shape: (1, seq_len)

        causal = causal_mask_tf(seq_len) # Shape: (1, seq_len, seq_len)

        decoder_mask = tf.logical_and(decoder_padding_mask, causal)
        decoder_mask = tf.cast(decoder_mask, tf.int64) # Shape: (1, seq_len, seq_len)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": x['src_text'],
            "tgt_text": x['tgt_text'],
        }

    train_ds = train_ds.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Batch and Prefetch
    train_dataloader = train_ds.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_dataloader = val_ds.batch(1).prefetch(tf.data.AUTOTUNE) # Val batch size is 1 for sentence-by-sentence evaluation

    # --- End of the tf.data preprocessing pipeline ---

    # Calculate the maximum length of sentences in the source and target sets (for informational purposes)
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        config['seq_len'], 
        config['seq_len'],
        config['d_model']
    )

    return model

def train_model(config):
    # Device detection
    gpus = tf.config.list_physical_devices('GPU')
    mps = tf.config.list_physical_devices('MPS')
    if gpus:
        device = '/GPU:0'
        print(f"Using device: GPU - {gpus[0].name}")
    elif mps:
        device = '/MPS:0'
        print("Using device: MPS")
    else:
        device = '/CPU:0'
        print("Using device: CPU")
        print("NOTE: If you have a GPU, consider configuring it for TensorFlow.")
        print("      On Windows with NVidia GPU, see: https://www.tensorflow.org/install/gpu")
        print("      On Mac (M1/M2), see: https://developer.apple.com/metal/tensorflow-plugin/")

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Load dataset and model
    train_dataset, val_dataset, tokenizer_src, tokenizer_tgt = get_ds(config)
    vocab_size_src = tokenizer_src.get_vocab_size()
    vocab_size_tgt = tokenizer_tgt.get_vocab_size()

    with tf.device(device):
        model = get_model(config, vocab_size_src, vocab_size_tgt)

    # TensorBoard
    writer = tf.summary.create_file_writer(config['experiment_name'])

    # Optimizer & loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'], epsilon=1e-9)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    # Restore checkpoint if exists
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=f"{config['datasource']}_{config['model_folder']}",
        max_to_keep=5
    )
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        ckpt_path = checkpoint_manager.latest_checkpoint
        if ckpt_path:
            checkpoint.restore(ckpt_path)
            print(f"Restored checkpoint from {ckpt_path}")
        else:
            print("No checkpoint found, starting from scratch.")
    else:
        print("No preload specified, starting from scratch.")

    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        print(f"Epoch {epoch:02d} ---------------------")
        # Progress bar
        batch_iterator = tqdm(train_dataset, desc=f"Processing Epoch {epoch:02d}")

        epoch_loss_avg = tf.keras.metrics.Mean()

        for batch in batch_iterator:
            encoder_input = batch['encoder_input']
            decoder_input = batch['decoder_input']
            label = batch['label']
            encoder_mask = batch['encoder_mask']
            decoder_mask = batch['decoder_mask']

            with tf.device(device):
                with tf.GradientTape() as tape:
                    # Forward pass
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    logits = model.project(decoder_output)

                    # Compute loss
                    loss = loss_fn(label, logits)
                    # Mask loss where label == PAD
                    mask = tf.cast(tf.not_equal(label, tokenizer_src.token_to_id('[PAD]')), dtype=loss.dtype)
                    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

                # Backward & optimize
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                epoch_loss_avg.update_state(loss)
                global_step += 1

            # Log
            batch_iterator.set_postfix({"loss": f"{loss.numpy():6.3f}"})
            with writer.as_default():
                tf.summary.scalar('train loss', loss, step=global_step)

        # Validation
        run_validation(model, val_dataset, tokenizer_src, tokenizer_tgt, config['seq_len'], device, print, global_step, writer)

        # Save checkpoint
        checkpoint_manager.save()
        print(f"Checkpoint saved at epoch {epoch:02d}")

    print("Training complete ✅")

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    with tf.device(device):
        # Encoder output (precomputed)
        encoder_output = model.encode(source, source_mask)

        # Start decoder input with SOS
        decoder_input = tf.constant([[sos_idx]], dtype=source.dtype)

        for _ in range(max_len):
            # build causal mask
            seq_len = decoder_input.shape[1]
            decoder_mask = causal_mask_tf(seq_len)

            # Forward through decoder & projection
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            logits = model.project(out[:, -1])

            next_word = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()[0]

            # Append next_word to decoder_input
            next_word_tensor = tf.constant([[next_word]], dtype=source.dtype)
            decoder_input = tf.concat([decoder_input, next_word_tensor], axis=1)

            if next_word == eos_idx:
                break

        return decoder_input[0]

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval() if hasattr(model, "eval") else None  # mostly for Keras it’s not needed

    count = 0
    source_texts, expected, predicted = [], [], []

    # get console width
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with tf.device(device):
        for batch in validation_ds:
            count += 1

            encoder_input = batch["encoder_input"]
            encoder_mask = batch["encoder_mask"]

            # check batch size == 1
            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

            decoded_seq = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device
            )

            # Decode to text
            source_text = batch["src_text"][0].numpy().decode('utf-8')
            target_text = batch["tgt_text"][0].numpy().decode('utf-8')
            predicted_text = tokenizer_tgt.decode(decoded_seq.numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(predicted_text)

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{predicted_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    # Metrics
    if writer:
        with writer.as_default():
            # Character error rate
            cer = character_accuracy_tf(expected, predicted)
            tf.summary.scalar('validation cer', cer, step=global_step)

            # Word error rate (TensorFlow Addons doesn’t have WER; implement manually if needed)
            wer = tf.reduce_mean([tf.edit_distance(tf.strings.unicode_split(p, 'UTF-8'),
                                                            tf.strings.unicode_split(t, 'UTF-8')).numpy()
                                  for p, t in zip(predicted, expected)])
            tf.summary.scalar('validation wer', wer, step=global_step)

            # BLEU score
            # bleu = tf.metrics.bleu_score([p.split() for p in predicted],
            #                               [[t.split()] for t in expected])[0]
            # tf.summary.scalar('validation BLEU', bleu, step=global_step)


def character_accuracy_tf(expected_strings, predicted_strings):
    """
    Calculates character accuracy using core TensorFlow operations.

    Args:
        expected_strings: A tf.Tensor of strings or a list of strings representing
                          the true (reference) sequences.
        predicted_strings: A tf.Tensor of strings or a list of strings representing
                           the predicted sequences.

    Returns:
        A tf.Tensor representing the character accuracy.
    """
    # Ensure inputs are TensorFlow tensors of string type
    expected_strings = tf.convert_to_tensor(expected_strings, dtype=tf.string)
    predicted_strings = tf.convert_to_tensor(predicted_strings, dtype=tf.string)

    # 1. Tokenize to characters
    expected_chars = tf.strings.unicode_split(expected_strings, 'UTF-8')
    predicted_chars = tf.strings.unicode_split(predicted_strings, 'UTF-8')

    # 2. Calculate edit distance (number of edits)
    # `normalize=False` means we get the raw edit count
    edit_distances = tf.edit_distance(predicted_chars, expected_chars, normalize=False)

    # 3. Calculate character lengths of the *expected* strings
    # We use float32 for division
    expected_char_lengths = tf.cast(tf.strings.length(expected_strings, unit='UTF8_CHAR'), dtype=tf.float32)

    # Handle cases where expected_char_lengths might be zero to avoid division by zero (NaN)
    # If the expected string is empty, edit distance is 0, length is 0. CER should be 0.
    # We use tf.where to conditionally set CER to 0 if length is 0.
    character_error_rate = tf.where(tf.equal(expected_char_lengths, 0),
                                    0.0,  # If length is 0, CER is 0 (or could be 1 depending on desired behavior for empty strings)
                                    edit_distances / expected_char_lengths)

    # Calculate the mean CER across the batch
    mean_character_error_rate = tf.reduce_mean(character_error_rate)

    # 4. Calculate Character Accuracy: 1 - CER
    char_accuracy = 1.0 - mean_character_error_rate

    return char_accuracy

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
