package com.infinitestack.javabert;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Responsible for loading the BERT model in ONNX format and generating embeddings for a given text.
 */
public class BertEmbeddingEngine implements EmbeddingEngine {

    private final Predictor<String, float[]> predictor;

    /**
     * Constructor that loads the ONNX model and configures the Predictor.
     *
     * @param modelPath Path to the ONNX BERT model file
     * @param vocabPath Path to the BERT vocab.txt file
     */
    public BertEmbeddingEngine(Path modelPath, Path vocabPath) throws IOException, ModelNotFoundException, MalformedModelException {
        // Load the vocabulary
        Vocabulary vocab = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(vocabPath)
                .optUnknownToken("[UNK]")
                .build();

        // Instantiate a simple BERT tokenizer
        BertTokenizer tokenizer = new BertTokenizer();

        // Build Criteria with our custom translator
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(modelPath)
                .optTranslator(new BertEmbedTranslator(tokenizer, vocab))
                .optEngine("OnnxRuntime")
                .build();

        // Load the model and create the predictor
        ZooModel<String, float[]> model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    /**
     * Generates and returns the embedding (float array) for a given text.
     *
     * @param text Input text
     * @return Float array representing the embedding (e.g., [CLS] token)
     */
    public float[] getEmbedding(String text) throws TranslateException {
        return predictor.predict(text);
    }

    /**
     * Custom translator that tokenizes the text and extracts the embedding from the [CLS] token.
     */
    private static class BertEmbedTranslator implements Translator<String, float[]> {
        private final BertTokenizer tokenizer;
        private final Vocabulary vocab;
        private final int maxSeqLength = 512;

        public BertEmbedTranslator(BertTokenizer tokenizer, Vocabulary vocab) {
            this.tokenizer = tokenizer;
            this.vocab = vocab;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            // Tokenize the input
            List<String> tokens = tokenizer.tokenize(input);

            // Debug: Display tokens for the input
            System.out.println("DEBUG - Tokens for input: " + input);
            System.out.println("DEBUG - List of tokens: " + tokens);

            // Optional: Display only the first 10 tokens or a summary
            // System.out.println("DEBUG - First tokens: " + tokens.subList(0, Math.min(10, tokens.size())));

            // Check for unknown tokens [UNK]
            if (tokens.contains("[UNK]")) {
                System.out.println("WARNING: [UNK] token generated for this input!");
            }

            // Check if truncation occurred at 512 tokens
            if (tokens.size() > maxSeqLength) {
                tokens = tokens.subList(0, maxSeqLength);
                System.out.println("DEBUG - Truncated to the first " + maxSeqLength + " tokens.");
            }

            // Convert tokens to IDs
            long[] inputIds = tokens.stream().mapToLong(token -> vocab.getIndex(token)).toArray();

            // Create attention_mask and token_type_ids
            long[] attentionMask = new long[inputIds.length];
            Arrays.fill(attentionMask, 1);
            long[] tokenTypeIds = new long[inputIds.length];

            // Create NDArrays
            NDArray inputIdsArr = ctx.getNDManager().create(inputIds);
            NDArray attentionArr = ctx.getNDManager().create(attentionMask);
            NDArray tokenTypeArr = ctx.getNDManager().create(tokenTypeIds);

            // Assign names
            inputIdsArr.setName("input_ids");
            attentionArr.setName("attention_mask");
            tokenTypeArr.setName("token_type_ids");

            return new NDList(inputIdsArr, attentionArr, tokenTypeArr);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray lastHiddenState = list.get(0); // Shape can be [batch, seq_length, hidden_size] or [seq_length, hidden_size]
            long rank = lastHiddenState.getShape().dimension();

            if (rank == 3) {
                // [1, seq_length, hidden_size]
                NDArray clsEmb = lastHiddenState.get(":, 0, :");
                return clsEmb.toFloatArray();
            } else if (rank == 2) {
                // [seq_length, hidden_size]
                NDArray clsEmb = lastHiddenState.get(0);
                return clsEmb.toFloatArray();
            } else {
                throw new IllegalArgumentException("Unexpected output format: " + lastHiddenState.getShape());
            }
        }
    }
}
