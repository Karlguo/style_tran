from tokenization import read_text_data_bpe


data_path="/home/zilu.gzl/data/formal_informal/new/single/test.0"
vocab_path="/home/zilu.gzl/data/formal_informal/new/vocab"

max_encoder_seq_length, char2idx, idx2char, encoder_input_data, decoder_input_data, decoder_output_data = read_text_data_bpe(data_path=data_path, vocab_path=vocab_path)
print (max_encoder_seq_length)
print (len(char2idx))
print (list(char2idx.items())[0:3])
print (len(idx2char))
print (list(idx2char.items())[0:3])
print (encoder_input_data.shape)
print (encoder_input_data[1][0:30])
print (decoder_input_data.shape)
print (decoder_output_data.shape)
