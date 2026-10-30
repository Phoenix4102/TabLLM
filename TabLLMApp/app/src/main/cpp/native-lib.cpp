#include <chrono>
#include <cstdio> // for snprintf
#include <jni.h>
#include <string>
#include <android/log.h>
#include <mutex>
#include <memory>
#include <vector>
#include <stdexcept>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "TabLLM", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "TabLLM", __VA_ARGS__)

// Include the entire engine as a single translation unit (Unity Build)
#include "engine.cpp"

static std::unique_ptr<Engine> g_engine;
static std::unique_ptr<Tokenizer> g_tokenizer;
static std::mutex g_mutex;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_tabllm_MainActivity_initEngine(
    JNIEnv* env, jobject, jstring modelPath, jstring tokenizerPath) {
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    const char* m_path = env->GetStringUTFChars(modelPath, nullptr);
    const char* t_path = env->GetStringUTFChars(tokenizerPath, nullptr);
    
    try {
        g_engine = std::make_unique<Engine>(m_path);
        g_tokenizer = std::make_unique<Tokenizer>();
        g_tokenizer->load(t_path);
        LOGI("Engine initialized successfully.");
        env->ReleaseStringUTFChars(modelPath, m_path);
        env->ReleaseStringUTFChars(tokenizerPath, t_path);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("Failed to init engine: %s", e.what());
        env->ReleaseStringUTFChars(modelPath, m_path);
        env->ReleaseStringUTFChars(tokenizerPath, t_path);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_tabllm_MainActivity_generate(
    JNIEnv* env, jobject, jstring prompt, jint maxTokens, jfloat temp, jint topK, jfloat topP, jfloat repPenalty, jobject callback) {
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    // Get Kotlin callback methods
    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
    jmethodID onComplete = env->GetMethodID(callbackClass, "onComplete", "(Ljava/lang/String;)V");

    if (!g_engine || !g_tokenizer) {
        jstring err = env->NewStringUTF("Error: Engine not initialized.");
        env->CallVoidMethod(callback, onComplete, err);
        return;
    }
    
    const char* p_str = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_text(p_str);
    env->ReleaseStringUTFChars(prompt, p_str);
    
    try {
        std::string final_prompt = "<|im_start|>user\n" + prompt_text + "<|im_end|>\n<|im_start|>assistant\n";
        
        std::vector<int> input = g_tokenizer->encode(final_prompt);
        if (input.empty()) input.push_back(0);
        
        g_engine->pos = 0; 
        
        auto t_start = std::chrono::high_resolution_clock::now();

        const float* logits = nullptr;
        if (input.size() == 1) {
            logits = g_engine->forward(input[0]);
        } else {
            logits = g_engine->prefill(input);
        }

        auto t_prompt_end = std::chrono::high_resolution_clock::now();
        
        std::random_device rd;
        std::mt19937 rng(rd());
        
        std::vector<int> generated;
        int next_token = sample(logits, g_engine->cfg.vocab_size, temp, topK, topP, repPenalty, generated, rng);
        
        int stop_token = 2; 
        
        for (int step = 0; step < maxTokens && g_engine->pos < g_engine->cfg.ctx_len; ++step) {
            if (next_token < 0 || next_token == stop_token) break;
            generated.push_back(next_token);
            
            // --- STREAMING MAGIC ---
            // Decode the current history and push it to the UI
            std::string current_text = g_tokenizer->decode(generated);
            jstring jText = env->NewStringUTF(current_text.c_str());
            env->CallVoidMethod(callback, onToken, jText);
            env->DeleteLocalRef(jText);
            // ---------------------
            
            if (g_engine->pos >= g_engine->cfg.ctx_len) break;
            
            logits = g_engine->forward(next_token);
            next_token = sample(logits, g_engine->cfg.vocab_size, temp, topK, topP, repPenalty, generated, rng);
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();

        long long prompt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prompt_end - t_start).count();
        long long gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_prompt_end).count();
        
        double gen_s = gen_ms / 1000.0;
        double tok_per_sec = gen_s > 0.0 ? generated.size() / gen_s : 0.0;

        std::string final_result = g_tokenizer->decode(generated);
        
                char stats[512];
        snprintf(stats, sizeof(stats), 
                 "\n\n--- Performance Stats ---\n"
                 "Prompt Eval (TTFT): %lld ms\n"
                 "Generation: %lld ms (%.1f tok/s)\n"
                 "Total Time: %lld ms\n"
                 "Tokens Generated: %zu",
                 prompt_ms, gen_ms, tok_per_sec, prompt_ms + gen_ms, generated.size());
        
        final_result += stats;
        
        jstring jFinal = env->NewStringUTF(final_result.c_str());
        env->CallVoidMethod(callback, onComplete, jFinal);
        env->DeleteLocalRef(jFinal);
        
    } catch (const std::exception& e) {
        LOGE("Generation failed: %s", e.what());
        std::string err = "Error: ";
        err += e.what();
        jstring jErr = env->NewStringUTF(err.c_str());
        env->CallVoidMethod(callback, onComplete, jErr);
        env->DeleteLocalRef(jErr);
    }
}