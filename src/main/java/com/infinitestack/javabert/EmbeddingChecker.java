package com.infinitestack.javabert;

import ai.djl.translate.TranslateException;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Class responsible for:
 * 1) Receiving a query,
 * 2) Generating the query embedding (via EmbeddingEngine),
 * 3) Comparing with stored document embeddings,
 * 4) Returning a ranking of the most relevant documents.
 */
public class EmbeddingChecker {

    private final EmbeddingEngine engine;
    private final List<DocumentEmbedding> storedDocs;

    /**
     * Constructor that receives:
     *  - The engine responsible for generating embeddings,
     *  - The list of documents (or chunks) with precomputed embeddings.
     */
    public EmbeddingChecker(EmbeddingEngine engine, List<DocumentEmbedding> storedDocs) {
        this.engine = engine;
        this.storedDocs = storedDocs;
    }

    /**
     * Receives a query, generates its embedding, and creates a ranking of the most similar documents.
     *
     * @param query Input string (e.g., user query)
     * @return List of DocumentRanking, sorted by similarity (descending).
     */
    public List<DocumentRanking> check(String query) throws TranslateException {
        // 1) Generate the query embedding
        float[] queryEmbedding = engine.getEmbedding(query);

        // 2) For each stored document, calculate the similarity
        List<DocumentRanking> ranking = new ArrayList<>();
        for (DocumentEmbedding docEmb : storedDocs) {
            float sim = cosineSimilarity(queryEmbedding, docEmb.getEmbedding());
            ranking.add(new DocumentRanking(docEmb, sim));
        }

        // 3) Sort the list from highest to lowest similarity
        ranking.sort(Comparator.comparing(DocumentRanking::getScore).reversed());

        return ranking;
    }

    /**
     * Example function to calculate cosine similarity between two float[] vectors.
     */
    private float cosineSimilarity(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors have different dimensions!");
        }

        // Dot product
        float dot = 0f;
        // Norm of v1
        float normV1 = 0f;
        // Norm of v2
        float normV2 = 0f;

        for (int i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            normV1 += v1[i] * v1[i];
            normV2 += v2[i] * v2[i];
        }

        // Avoid division by zero
        if (normV1 == 0 || normV2 == 0) {
            return 0f;
        }

        return (float) (dot / (Math.sqrt(normV1) * Math.sqrt(normV2)));
    }
}
