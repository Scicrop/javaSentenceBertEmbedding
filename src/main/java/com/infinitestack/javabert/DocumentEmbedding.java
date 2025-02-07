package com.infinitestack.javabert;

public class DocumentEmbedding {
    private String docId;
    private float[] embedding;

    public DocumentEmbedding(String docId, float[] embedding) {
        this.docId = docId;
        this.embedding = embedding;
    }

    public String getDocId() {
        return docId;
    }

    public float[] getEmbedding() {
        return embedding;
    }
}
