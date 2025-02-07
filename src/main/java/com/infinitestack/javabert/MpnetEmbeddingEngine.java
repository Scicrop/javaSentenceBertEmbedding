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
 * Responsible for loading the all-mpnet-base-v2 model (exported to ONNX) and generating embeddings for a text.
 * In this version, we only use the inputs "input_ids" and "attention_mask" as expected by the model.
 * The output is processed via mean pooling, as is standard for Sentence-BERT.
 */
public class MpnetEmbeddingEngine implements EmbeddingEngine {

    private final Predictor<String, float[]> predictor;

    /**
     * Constructor that loads the ONNX model and configures the Predictor.
     *
     * @param modelPath Path to the ONNX file of the all-mpnet-base-v2 model
     * @param vocabPath Path to the corresponding vocab.txt file
     */
    public MpnetEmbeddingEngine(Path modelPath, Path vocabPath)
            throws IOException, ModelNotFoundException, MalformedModelException {
        // Load the vocabulary
        Vocabulary vocab = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(vocabPath)
                .optUnknownToken("[UNK]")
                .build();

        // Instantiate the tokenizer (here we use BertTokenizer; adapt if the model uses another)
        BertTokenizer tokenizer = new BertTokenizer();

        // Build Criteria with our custom translator for MPNet
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(modelPath)
                .optTranslator(new MpnetEmbedTranslator(tokenizer, vocab))
                .optEngine("OnnxRuntime")
                .build();

        // Load the model and create the predictor
        ZooModel<String, float[]> model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    /**
     * Generates and returns the embedding (float array) for a text.
     *
     * @param text Input text.
     * @return Float array representing the embedding (resulting from mean pooling of tokens).
     */
    public float[] getEmbedding(String text) throws TranslateException {
        return predictor.predict(text);
    }

    /**
     * Custom translator for all-mpnet-base-v2.
     * It tokenizes the text and, in the post-processing step, applies mean pooling to generate the sentence embedding.
     * Only the inputs "input_ids" and "attention_mask" are sent, as expected by the model.
     */
    private static class MpnetEmbedTranslator implements Translator<String, float[]> {
        private final BertTokenizer tokenizer;
        private final Vocabulary vocab;
        private final int maxSeqLength = 512;

        public MpnetEmbedTranslator(BertTokenizer tokenizer, Vocabulary vocab) {
            this.tokenizer = tokenizer;
            this.vocab = vocab;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            // Tokenize the text
            List<String> tokens = tokenizer.tokenize(input);
            System.out.println("DEBUG - Tokens for input: " + input);
            System.out.println("DEBUG - List of tokens: " + tokens);

            // Truncate the sequence if it exceeds the maximum length
            if (tokens.size() > maxSeqLength) {
                tokens = tokens.subList(0, maxSeqLength);
                System.out.println("DEBUG - Truncated to the first " + maxSeqLength + " tokens.");
            }

            // Convert tokens to IDs
            long[] inputIds = tokens.stream().mapToLong(token -> vocab.getIndex(token)).toArray();

            // Create the attention mask: 1 for each token
            long[] attentionMask = new long[inputIds.length];
            Arrays.fill(attentionMask, 1);

            // Create 1D NDArrays (Predictor will handle batching automatically)
            NDArray inputIdsArray = ctx.getNDManager().create(inputIds);
            NDArray attentionMaskArray = ctx.getNDManager().create(attentionMask);

            // Assign expected names for the inputs
            inputIdsArray.setName("input_ids");
            attentionMaskArray.setName("attention_mask");

            // Return NDArrays in an NDList (only 2 inputs)
            NDList ndList = new NDList();
            ndList.add(inputIdsArray);
            ndList.add(attentionMaskArray);
            return ndList;
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray lastHiddenState = list.get(0);
            // For all-mpnet-base-v2, apply mean pooling along the sequence dimension.
            if (lastHiddenState.getShape().dimension() == 3) {
                // lastHiddenState has shape [batch, seq_length, hidden_size]
                NDArray meanEmb = lastHiddenState.mean(new int[]{1});
                return meanEmb.toFloatArray();
            } else if (lastHiddenState.getShape().dimension() == 2) {
                // If there is no batch dimension, assume [seq_length, hidden_size] and average along the first dimension
                NDArray meanEmb = lastHiddenState.mean(new int[]{0});
                return meanEmb.toFloatArray();
            } else {
                throw new IllegalArgumentException("Unexpected output format: " + lastHiddenState.getShape());
            }
        }
    }
}
