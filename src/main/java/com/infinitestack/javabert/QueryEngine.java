package com.infinitestack.javabert;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class QueryEngine {

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        // Check if we have 2 arguments: the query and the directory with embeddings
        if (args.length < 2) {
            System.err.println("Usage: java -jar QueryEngine.jar \"your query\" /path/to/onnx_model /path/to/embeddings/directory");
            System.exit(1);
        }

        String query = args[0].toLowerCase();
        String embeddingsDir = args[2];

        // Load document embeddings from .json files
        List<DocumentEmbedding> storedDocs = loadEmbeddingsFromDirectory(embeddingsDir);

        // Fixed or configurable paths to the MPNet model and vocabulary
        // (You can also pass these via args or environment variables)
        String modelPath = args[1]+"/model.onnx";
        String vocabPath = args[1]+"/vocab.txt";

        // Instantiate the MpnetEmbeddingEngine
        MpnetEmbeddingEngine engine = new MpnetEmbeddingEngine(
                Paths.get(modelPath),
                Paths.get(vocabPath)
        );

        // Create the EmbeddingChecker using the engine and the list of documents
        EmbeddingChecker checker = new EmbeddingChecker(engine, storedDocs);

        // Execute the check method to obtain the ranking
        List<DocumentRanking> ranking = checker.check(query);

        if (ranking.isEmpty()) {
            System.out.println("No documents found in the embeddings directory.");
            return;
        }

        // Get the top-1 (first in the ranking)
        DocumentRanking topDoc = ranking.get(0);

        System.out.println("Query: " + query);
        System.out.println("Most related document: " + topDoc.getDocument().getDocId());
        System.out.printf("Cosine similarity: %.4f%n", topDoc.getScore());
    }

    /**
     * Reads all .json files from the directory, deserializing them as EmbeddingJson (filename + embeddings).
     * Then, creates a DocumentEmbedding for each object.
     */
    private static List<DocumentEmbedding> loadEmbeddingsFromDirectory(String dirPath) throws IOException {
        List<DocumentEmbedding> docs = new ArrayList<>();
        File dir = new File(dirPath);
        if (!dir.isDirectory()) {
            System.err.println("The provided path is not a valid directory: " + dirPath);
            return docs;
        }

        File[] files = dir.listFiles((f, name) -> name.toLowerCase().endsWith(".json"));
        if (files == null) {
            System.err.println("No JSON files found in: " + dirPath);
            return docs;
        }

        ObjectMapper mapper = new ObjectMapper();

        for (File file : files) {
            // Read the EmbeddingJson (filename + embeddings)
            EmbeddingJson ej = mapper.readValue(file, EmbeddingJson.class);
            // Create DocumentEmbedding using the filename from the JSON as docId
            DocumentEmbedding de = new DocumentEmbedding(ej.filename, ej.embeddings);
            docs.add(de);
        }

        return docs;
    }
}
