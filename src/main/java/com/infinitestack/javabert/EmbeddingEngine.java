package com.infinitestack.javabert;

import ai.djl.translate.TranslateException;

public interface EmbeddingEngine {
    float[] getEmbedding(String text) throws TranslateException;
}
