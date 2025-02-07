package com.infinitestack.javabert;

public class DocumentRanking {
    private DocumentEmbedding document;
    private float score;

    public DocumentRanking(DocumentEmbedding document, float score) {
        this.document = document;
        this.score = score;
    }

    public DocumentEmbedding getDocument() {
        return document;
    }

    public float getScore() {
        return score;
    }
}
