package com.infinitestack.javabert;

public class EmbeddingJson {
    public String filename;
    public float[] embeddings;

    public EmbeddingJson() {
    }

    public EmbeddingJson(String filename, float[] embeddings) {
        this.filename = filename;
        this.embeddings = embeddings;
    }

    public String getFilename() {
        return filename;
    }

    public float[] getEmbeddings() {
        return embeddings;
    }
}
