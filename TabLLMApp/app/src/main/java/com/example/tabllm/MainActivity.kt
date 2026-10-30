package com.example.tabllm
import com.example.tabllm.R  // <--- ADD THIS LINE

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.tabllm.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    // Interface for C++ to push live tokens to Kotlin
    interface StreamCallback {
        fun onToken(text: String)
        fun onComplete(text: String)
    }

    companion object {
        init {
            System.loadLibrary("tabllm_jni")
        }
    }

    private external fun initEngine(modelPath: String, tokenizerPath: String): Boolean
    private external fun generate(prompt: String, maxTokens: Int, temp: Float, topK: Int, topP: Float, repPenalty: Float, callback: StreamCallback)

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val filesDir = getExternalFilesDir(null)
        
        val tllmFiles = filesDir?.listFiles { _, name -> name.endsWith(".tllm") }?.map { it.name } ?: emptyList()
        val tokenizerFile = File(filesDir, "tokenizer.json")

        if (tllmFiles.isEmpty() || !tokenizerFile.exists()) {
            binding.textViewOutput.text = "Error: Models or tokenizer not found.\nPush .tllm files and tokenizer.json via adb to:\n${filesDir?.absolutePath}"
            binding.buttonGenerate.isEnabled = false
            return
        }

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, tllmFiles)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModels.adapter = adapter

        binding.textViewOutput.text = "Ready. Select a model, type a prompt, and tap Generate."
        binding.buttonGenerate.isEnabled = true

        binding.buttonGenerate.setOnClickListener {
            val prompt = binding.editTextPrompt.text.toString().trim()
            if (prompt.isEmpty()) {
                Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val maxTokens = binding.editMaxTokens.text.toString().toIntOrNull() ?: 128
            val temp = binding.editTemp.text.toString().toFloatOrNull() ?: 0.7f
            val topK = binding.editTopK.text.toString().toIntOrNull() ?: 40
            val topP = binding.editTopP.text.toString().toFloatOrNull() ?: 0.9f
            val repPenalty = binding.editRepPenalty.text.toString().toFloatOrNull() ?: 1.1f

            val selectedModel = binding.spinnerModels.selectedItem as String
            val selectedModelFile = File(filesDir, selectedModel)

            binding.buttonGenerate.isEnabled = false
            binding.textViewOutput.text = "Loading model..."

            // Create the callback to receive live streaming text
            val callback = object : StreamCallback {
                override fun onToken(text: String) {
                    runOnUiThread {
                        binding.textViewOutput.text = text
                    }
                }

                override fun onComplete(text: String) {
                    runOnUiThread {
                        binding.textViewOutput.text = text
                        binding.buttonGenerate.isEnabled = true
                    }
                }
            }

            lifecycleScope.launch {
                withContext(Dispatchers.IO) {
                    val success = initEngine(selectedModelFile.absolutePath, tokenizerFile.absolutePath)
                    if (!success) {
                        withContext(Dispatchers.Main) {
                            binding.textViewOutput.text = "Error: Failed to initialize ${selectedModel}."
                            binding.buttonGenerate.isEnabled = true
                        }
                        return@withContext
                    }
                    
                    // Start generation. C++ will push tokens to 'callback' live.
                    generate(prompt, maxTokens, temp, topK, topP, repPenalty, callback)
                }
            }
        }
    }
}