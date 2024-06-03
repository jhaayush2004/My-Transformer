
# Attention is All You Need !!!

What do you actually need from this world ? Attention is All You Need!
Let's take a look of the game changer architecture of Transformer .



![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/Screenshot%202024-06-03%20125704.png)


##  Motivation for Transitioning from RNNs to Transformers

1. **Difficulty in Learning Long-range Dependencies:** RNNs struggle to effectively capture dependencies between distant positions in a sequence due to their sequential nature and vanishing gradient problem.
  
2. **Limited Contextual Understanding:** Traditional RNN architectures like vanilla RNNs and LSTMs have difficulty retaining and utilizing contextual information from distant parts of the sequence, leading to suboptimal performance in tasks requiring long-range dependencies.

3. **Inefficient Training:** Training RNNs on long sequences requires processing each element sequentially, making it computationally expensive and time-consuming, especially for large datasets.

4. **Challenges in Parallelization:** RNNs inherently process sequences sequentially, limiting their parallelization potential and hindering scalability on modern hardware architectures like GPUs and TPUs.

5. **Prone to Gradient Vanishing and Exploding:** RNNs are prone to the vanishing and exploding gradient problems, which can severely affect the model's ability to learn and generalize from data over long sequences.
##  Motivation for Transitioning from RNNs to Transformers

1. **Difficulty in Learning Long-range Dependencies:** RNNs struggle to effectively capture dependencies between distant positions in a sequence due to their sequential nature and vanishing gradient problem.
  
2. **Limited Contextual Understanding:** Traditional RNN architectures like vanilla RNNs and LSTMs have difficulty retaining and utilizing contextual information from distant parts of the sequence, leading to suboptimal performance in tasks requiring long-range dependencies.

3. **Inefficient Training:** Training RNNs on long sequences requires processing each element sequentially, making it computationally expensive and time-consuming, especially for large datasets.

4. **Challenges in Parallelization:** RNNs inherently process sequences sequentially, limiting their parallelization potential and hindering scalability on modern hardware architectures like GPUs and TPUs.

5. **Prone to Gradient Vanishing and Exploding:** RNNs are prone to the vanishing and exploding gradient problems, which can severely affect the model's ability to learn and generalize from data over long sequences.
##  Motivation for Transitioning from RNNs to Transformers

1. **Difficulty in Learning Long-range Dependencies:** RNNs struggle to effectively capture dependencies between distant positions in a sequence due to their sequential nature and vanishing gradient problem.
  
2. **Limited Contextual Understanding:** Traditional RNN architectures like vanilla RNNs and LSTMs have difficulty retaining and utilizing contextual information from distant parts of the sequence, leading to suboptimal performance in tasks requiring long-range dependencies.

3. **Inefficient Training:** Training RNNs on long sequences requires processing each element sequentially, making it computationally expensive and time-consuming, especially for large datasets.

4. **Challenges in Parallelization:** RNNs inherently process sequences sequentially, limiting their parallelization potential and hindering scalability on modern hardware architectures like GPUs and TPUs.

5. **Prone to Gradient Vanishing and Exploding:** RNNs are prone to the vanishing and exploding gradient problems, which can severely affect the model's ability to learn and generalize from data over long sequences.
## Embeddings
Embeddings encode information about the individual tokens and their positions in the sequence, providing the Transformer model with the necessary contextual information to perform tasks such as machine translation or text classification.
## Positional Encoding
- **Positional Embedding Significance:** In the absence of recurrence and convolutional structures within the Transformer model, it becomes imperative to incorporate positional information into the embeddings to exploit the sequence's inherent order.

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/PE%20image.png)
  
- **Encoding Scheme:** The positional embedding function, denoted as \( PE_{(pos, i)} \), combines sine and cosine functions, modulated by positional indices and embedding dimensions. The formulation is designed to facilitate the model's understanding of relative positions within the sequence.


![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/PE%20sine%20cos%20graph.png)

- **Motivation for Sinusoidal Function:** The authors selected the sinusoidal embedding function due to its potential to enable the model to effectively learn relative positional attentiveness. Specifically, the sinusoidal nature allows for linear representations of positional relationships, crucial for contextual understanding.

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/sine%20cosine%20PE.png)

- **Comparative Analysis:** Experimentation between sinusoidal embeddings and learned positional embeddings yielded nearly identical outcomes. Despite this, the sinusoidal approach was favored for its speculated ability to generalize to longer sequence lengths beyond those encountered during training.

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/sine%20cosine%20PE.png)

- **Flexibility in Embedding Strategies:** While the sinusoidal embedding method is prevalent, users retain the flexibility to explore alternative positional embedding strategies, including traditional word embedding techniques, to suit specific model requirements.
## Scaled-Dot-Product Attention

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/scaled%20dot%20product.png)

- **Retrieval Systems Analogy:** The key/value/query mechanism in Transformers parallels retrieval systems like YouTube searches. Your search query (query) is compared against keys (video titles, descriptions) to retrieve relevant videos (values).

- **Value Retrieval:** In Transformers, a value \( v_i \) is retrieved for a query \( q \) by evaluating its correlation with a corresponding key \( k_i \). This forms the core of the attention mechanism, enabling context-aware processing.

- **Tensor Score Interpretation:** The tensor score indicates the correlation between each word in a sequence. It guides subsequent attention-based computations, providing insights into inter-word relationships.

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/attension%20score%20matrix.png)

- During translation tasks, like from Kannada to English, the model follows a sequential approach akin to humans. The encoder sees the input sequence without masking, allowing it to consider future tokens. Meanwhile, the decoder, responsible for generating the target sequence, is masked during training to prevent it from accessing future information, ensuring a fair translation process. This principle applies universally to any input-target sequence pair, maintaining integrity in translation tasks.

- How to do Masking ???

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/mask1.png)

- So, we mask the future attention weights with -inf

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/scale%2Bmask.png)

- Then the softmax function makes them 0

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/softmax%20on%20mask.png)

- Now, when we multiply the attention weights with the value matrix, the future tokens for a particular token will get 0 weight and thus will not be able to see the future tokens

- If we mask the illegal connections as 0 after softmax, the sum won't be 1, so we mask the illegal connections as -inf before applying softmax, so after applying softmax the sum will be 1

## Multi-Head Attention
- Instead of performing a single Attention function with Queries, Keys and Values, the authors found that it is beneficial to linearly transform the Queries, Key and Values h times with different learned linear projections as shown in the above figure. Then they are concatenated and linear transformed again.

- Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/Multihead%20attention.png)

MultiHead(Q, K, V) = Concat(h_1, ..., h_h) where h_i = Scalled_Dot_Product_Attention(W_Q @ Q, W_K @ K, W_V @ V)

## Feed-Forward Layers
- 2 Feed-Forward Layers, 1st with a relu activation and the 2nd without any activation
- The dimensionality of input and output is d_model, the inner-layer has dimensionality d_model*3

## The Model
1. **Encoder**: The encoder in the model comprises five identical layers. Each layer includes self-attention mechanisms and position-wise fully connected feed-forward networks. This design enables the encoder to progressively extract hierarchical features from the input sequence, facilitating better understanding and representation of the input data.

2. **Decoder**: The decoder in the  model consists of five identical layers. These layers incorporate both self-attention mechanisms and cross-attention mechanisms over the encoder's output. With each layer, the decoder refines its understanding of the target sequence, leveraging information from the encoder's representations while attending to relevant parts of the input and output sequences.

![App Screenshot](https://github.com/jhaayush2004/Transformer/blob/main/Visulals/transformer_decoding_1.gif)

**Encoder-Decoder Attention (Cross-Attention):**
- Queries originate from the previous decoder layer.
- Keys and values, derived from the encoder's output, enable the decoder to attend to all positions in the input sequence. This mirrors the behavior of traditional encoder-decoder models with RNN-based attention mechanisms.

**Self-Attention in Encoder:**
- In the encoder, keys, queries, and values stem from the same source, typically the input or the preceding encoder layer. This allows each position in the encoder to attend to other positions within the input sequence.

**Self-Attention in Decoder:**
- To prevent the flow of future information (leftward) in the decoder, masking is applied. This ensures that the model does not have access to tokens it should predict later. In contrast to the encoder, which does not require masking, the decoder employs masking in self-attention.

**Regularization: Dropout:**
- Dropout is applied to the output of each sub-layer before normalization. It is also applied to the embeddings in both the encoder and decoder. This regularization technique helps prevent overfitting by randomly dropping units during training.

## Also Visit

- [Attention is All You Need 2017 Paper](https://www.github.com/octokatherine)
- [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [CodeEmporium](https://www.youtube.com/watch?v=QCJQG4DuHT0&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4)
