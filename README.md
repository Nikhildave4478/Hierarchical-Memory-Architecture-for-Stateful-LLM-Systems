<h1>🧠 Designing a Hierarchical Memory Architecture for Stateful LLM Systems</h1>
<h3>A Research-Oriented Exploration using LangChain, Ollama, and FAISS</h3>

<hr>

<h2>📖 Motivation</h2>

<p>
Large Language Models (LLMs) are stateless by design. While they generate context-aware responses within a single prompt window, they do not inherently possess persistent memory.
</p>

<p>This project explores:</p>
<ul>
<li>How to simulate multi-layer memory in LLM systems</li>
<li>How to manage context compression</li>
<li>How to prevent stale cache conflicts</li>
<li>How to design a retrieval-first architecture</li>
</ul>

<p>
The system evolved through multiple design iterations, solving real architectural problems encountered during experimentation.
</p>

<hr>

<h2>🔬 Research Journey</h2>

<h3>Phase 1 — Short-Term Memory</h3>
<p>
Initial implementation used <code>RunnableWithMessageHistory</code> to store conversation context.
</p>
<ul>
<li>Enabled short-term continuity</li>
<li>Simple implementation</li>
<li>Worked for brief conversations</li>
</ul>

<p><b>Problem:</b> Context window overflow and identity drift during long conversations.</p>

<hr>

<h3>Phase 2 — Rolling Summary Compression</h3>
<p>
When token count exceeded a threshold, conversation was summarized:
</p>

<pre><code>if total_tokens &gt; MAX_TOKENS:
    summarize_memory(session_id)
</code></pre>

<ul>
<li>Reduced context size</li>
<li>Maintained high-level conversation state</li>
</ul>

<p><b>New Problem:</b> Progressive summarization caused semantic drift and information loss.</p>

<hr>

<h3>Phase 3 — Long-Term Vector Memory (FAISS)</h3>
<p>
Accumulated summaries were archived into FAISS vector storage using sentence embeddings.
</p>

<ul>
<li>Preserved older context</li>
<li>Enabled semantic recall</li>
</ul>

<p><b>Critical Insight:</b> Storage alone does not enable memory. Retrieval must be integrated into generation.</p>

<hr>

<h3>Phase 4 — Retrieval Injection (RAG)</h3>

<pre><code>retrieved_memory = retrieve_long_term_memory(user_input)</code></pre>

<p>
Retrieved memory is injected into the prompt before LLM generation.
</p>

<p>This transformed the system into a Retrieval-Augmented Generation (RAG) architecture.</p>

<hr>

<h3>Phase 5 — Semantic Cache Introduction</h3>

<p>
To improve efficiency, an embedding-based semantic cache was added.
</p>

<ul>
<li>Cosine similarity thresholding</li>
<li>Response reuse for similar queries</li>
</ul>

<p><b>New Architectural Conflict:</b> Cache returned stale responses when memory state changed.</p>

<hr>

<h3>Phase 6 — Retrieval-First + Cache Invalidation</h3>

<p>
Pipeline redesigned:
</p>

<pre><code>User Input
   ↓
Long-Term Retrieval
   ↓
Augmented Input
   ↓
Semantic Cache Check
   ↓
LLM Generation</code></pre>

<p>
Additionally, cache invalidation occurs whenever long-term memory updates:
</p>

<pre><code>semantic_cache.pop(session_id, None)</code></pre>

<p>This eliminated memory-cache inconsistencies.</p>

<hr>

<h2>🏗 Final Architecture</h2>

<ul>
<li><b>L1:</b> Short-Term Working Memory</li>
<li><b>L2:</b> Rolling Summary Buffer</li>
<li><b>L3:</b> Long-Term Vector Memory (FAISS)</li>
<li><b>L4:</b> Semantic Cache Layer</li>
</ul>

<p>
This mirrors hierarchical memory systems found in cognitive models and CPU cache design.
</p>

<hr>

<h2>⚙️ Technology Stack</h2>

<ul>
<li>LangChain</li>
<li>Ollama (LLaMA 3)</li>
<li>FAISS Vector Store</li>
<li>HuggingFace Sentence Transformers</li>
<li>NumPy (Cosine Similarity)</li>
<li>Transformers Tokenizer</li>
</ul>

<hr>

<h2>🧪 Experimental Focus</h2>

<ul>
<li>Identity persistence across long sessions</li>
<li>Memory compression strategies</li>
<li>Cache-state consistency</li>
<li>Retrieval relevance thresholding</li>
<li>Context drift mitigation</li>
</ul>

<hr>

<h2>🚀 Running the System</h2>

<h3>Install Dependencies</h3>

<pre><code>pip install langchain langchain-community faiss-cpu transformers numpy</code></pre>

<h3>Pull LLaMA 3 via Ollama</h3>

<pre><code>ollama pull llama3</code></pre>

<h3>Run</h3>

<pre><code>python main.py</code></pre>

<hr>

<h2>🔭 Future Directions</h2>

<ul>
<li>Memory importance scoring</li>
<li>Memory decay models</li>
<li>Graph-based long-term memory</li>
<li>Conflict resolution between contradictory memories</li>
<li>Structured fact extraction</li>
</ul>
