package com.infinitestack.javabert;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

public class BertDataEmbedd {

    public static void main(String[] args) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {
        if (args.length < 2) {
            System.err.println("Usage: java -jar BertDataEmbedd.jar \"folder_with_text_to_embedd\" \"folder_with_onnx_model\" ");
            System.exit(1);
        }

        // Paths to the model and vocabulary
        String modelPath = args[1]+"/model.onnx";
        String vocabPath = args[1]+"/vocab.txt";

        // Instantiate the embedding engine
        BertEmbeddingEngine engine = new BertEmbeddingEngine(
                Paths.get(modelPath),
                Paths.get(vocabPath)
        );

        File inputDir = new File(args[0]);
        for (int i = 0; i < inputDir.list().length; i++) {
            String fileName = args[0] + "/" + inputDir.list()[i];
            System.out.println(fileName);
            String inputText = new String(Files.readAllBytes(Paths.get(fileName))).toLowerCase();

            // Generate the embedding
            float[] embedding = engine.getEmbedding(inputText);

            // Compute MD5 hash of the text for the output file name
            String md5Hash = computeMD5(inputText);

            // Create the JSON object to save
            EmbeddingJson embeddingJson = new EmbeddingJson(fileName, embedding);

            // Generate the .json file name based on MD5
            File outputFile = new File("/tmp/embeddings/" + md5Hash + ".json");

            // Save as JSON
            ObjectMapper mapper = new ObjectMapper();
            mapper.writeValue(outputFile, embeddingJson);

            // Output information
            System.out.println("Input text: " + inputText);
            System.out.println("Generated embedding (dimension: " + embedding.length + ")");
            System.out.println("Sample values: " + Arrays.toString(Arrays.copyOf(embedding, 10)) + "...");
            System.out.println("Embedding saved at: " + outputFile.getAbsolutePath());
        }
    }

    private static String computeMD5(String text) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(text.getBytes(StandardCharsets.UTF_8));
            return bytesToHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("MD5 not supported in this environment", e);
        }
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
