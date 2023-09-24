# LLM-Learning

A collection of interesting research papers, projects, and documents I've come across.

## Visio Diagrams (PDF Format)

- [Data to LLM Processes](Data%20to%20LLM%20Processes.pdf)
- [Foundation Models](Foundation%20Models.pdf)
- [LLM Stack](LLM%20Stack.pdf)
- [Multi_Models](Multi_Models.pdf)


## Table of Contents

1. [Research Papers](#research-papers)
2. [Projects](#projects)
3. [Links](#links)

---

## Research Papers

### 09/22/23:

- **Title**: Large Language Models for Compiler Optimization
- **Description**: The study investigates the use of Large Language Models for code optimization, using a 7B-parameter transformer model trained to enhance LLVM assembly code size. The model not only predicts optimized compiler options but also anticipates instruction counts pre and post-optimization, leading to a 3.0% improvement over traditional compilers and showcasing robust code reasoning abilities.
- **Link**: [Click here](https://arxiv.org/pdf/2309.07062.pdf)

- **Title**: PROMPT2MODEL: Generating Deployable Models from Natural Language Instructions
- **Description**: "Prompt2Model" is a proposed method that transforms natural language task descriptions into deployable special-purpose models, addressing the limitations of large language models (LLMs) which demand extensive computational resources and can be limited by APIs. In tests, models trained with Prompt2Model outperformed a prominent LLM by 20% while being up to 700 times smaller, also offering reliable performance assessments before deployment.
- **Link**: [Click here](https://arxiv.org/pdf/2309.07062.pdf)

- **Title**: Precise Zero-Shot Dense Retrieval without Relevance Labels
- **Description**: The paper introduces Hypothetical Document Embeddings (HyDE) as a solution to challenges in zero-shot dense retrieval systems. HyDE prompts an instruction-following language model to generate a hypothetical document based on a given query, which is then encoded into an embedding vector, enabling the retrieval of similar real documents from a corpus, filtering out any inaccuracies from the generated document. This approach significantly surpasses existing unsupervised dense retrievers and performs comparably to fine-tuned retrievers across multiple tasks and languages.
- **Link**: [Click here](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://aclanthology.org/2023.acl-long.99.pdf)


---

## Projects

- **Name**: Cleanlab
- **Description**: cleanlab helps you clean data and labels by automatically detecting issues in an ML dataset. To facilitate machine learning with messy, real-world data, this data-centric AI package uses your existing models to estimate dataset problems that can be fixed to train even better models.
- **Link**: [Click here](https://github.com/cleanlab/cleanlab).


- **Name**: Awesome-Multimodal-Large-Language-Models
- **Description**: A curated list of Multimodal Large Language Models (MLLMs), including datasets, multimodal instruction tuning, multimodal in-context learning, multimodal chain-of-thought, llm-aided visual reasoning, foundation models, and others. This list will be updated in real-time
- **Link**: [Click here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models).

- **Name**: DreamLLM
- **Description**: DreamLLM is a learning framework that first achieves versatile Multimodal Large Language Models (MLLMs) empowered with frequently overlooked synergy between multimodal comprehension and creation. DreamLLM operates on two fundamental principles. The first focuses on the generative modeling of both language and image posteriors by direct sampling in the raw multimodal space. Second, DreamLLM fosters the generation of raw, interleaved documents, modeling both text and image contents, along with unstructured layouts. DreamLLM is a zero-shot multimodal generalist capable of both comprehension and creation.
- **Link**: [Click here](https://github.com/RunpeiDong/DreamLLM).

- **Name**: NExT-GPT
- **Description**:This repository hosts the code, data, and model weight of NExT-GPT, the first end-to-end MM-LLM that perceives input and generates output in arbitrary combinations (any-to-any) of text, image, video, and audio and beyond.
- **Link**: [Click here](https://github.com/NExT-GPT/NExT-GPT).

- **Name**: LlaMa-Adapter
- **Description**: This repo proposes LLaMA-Adapter (V2), a lightweight adaption method for fine-tuning Instruction-following and Multi-modal LLaMA models 🔥.
- **Link**: [Click here](https://github.com/OpenGVLab/LLaMA-Adapter).

- **Name**: LargeLanguageModelsProjects
- **Description**: a collection of llama models in different deployment configurations.
- **Link**: [Click here](https://github.com/MuhammadMoinFaisal/LargeLanguageModelsProjects/blob/main/Chat%20with%20Multiple%20Documents/Chat_with_Multiple_Documents_Llama2_OpenAI_Chroma_comp.ipynb).

- **Name**: axolotl
- **Description**: Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.
- **Link**: [Click here](https://github.com/OpenAccess-AI-Collective/axolotl).

- **Name**: Prompt2Model - Generate Deployable Models from Instructions
- **Description**: "Prompt2Model" is a system that enables users to train specialized models from natural language prompts, streamlining the process of creating models for specific tasks. The system provides a guided demo that takes users through steps like data collection, text processing, and model fine-tuning, making it adaptable for various tasks by simply adjusting the initial prompt and other design variables.
- **Link**: [Click here](https://github.com/neulab/prompt2model/blob/main/prompt2model_demo.ipynb).

- **Name**: Prompt2Model - Generate Deployable Models from Instructions
- **Description**: LiteLLM manages:
Translating inputs to the provider's completion and embedding endpoints.
Guarantees consistent output, text responses will always be available at ['choices'][0]['message']['content'].
Exception mapping - common exceptions across providers are mapped to the OpenAI exception types.
- **Link**: [Click here](https://github.com/BerriAI/litellm).
---

## Links

- **Title**: Why Open Source AI Will Win
- **Description**: The article discusses the rising debate between open-source and closed-source AI models, emphasizing the potential and advantages of open-source AI in the future. The author argues that despite current challenges and misconceptions, open-source AI models offer greater control, transparency, flexibility, and potential for customization, making them a better choice for businesses and developers in the long run.
- **Link**: [Click here](https://varunshenoy.substack.com/p/why-open-source-ai-will-win)

- **Title**: RAG is more than just embedding search
- **Description**: The article discusses the potential of Retrieval Augmented Generation (RAG) in the context of large language models (LLMs). Using 'instructor' as a framework, it highlights how RAG can be enhanced beyond simple embeddings, emphasizing the importance of query understanding, information retrieval, and integrating LLMs with structured outputs for more precise and sophisticated AI-driven search experiences.
- **Link**: [Click here](https://jxnl.github.io/instructor/blog/2023/09/17/rag-is-more-than-just-embedding-search/)

- **Title**: Guide to Chroma DB | A Vector Store for Your Generative AI LLMs
- **Description**: The article provides a comprehensive guide to Chroma DB, an open-source Vector Database designed for storing and efficiently retrieving vector embeddings. Generative Large Language Models utilize vector embeddings, which are numerical representations of data (like text, images, etc.), to understand and process information. Chroma DB facilitates the storage and retrieval of these embeddings, enhancing tasks such as semantic search, recommendations, and chatbot responses. The guide demonstrates how to set up and use Chroma DB, including creating collections, adding documents, and querying relevant information.
- **Link**: [Click here](https://www.analyticsvidhya.com/blog/2023/07/guide-to-chroma-db-a-vector-store-for-your-generative-ai-llms/)

- **Title**: Building a Knowledge base for custom LLMs using Langchain, Chroma, and GPT4All
- **Description**: The blog post discusses how to build a knowledge base for custom Large Language Models (LLMs) using Langchain, Chroma, and GPT4All. The author, Anindyadeep, guides readers through creating an application named DocChat, where users can upload and interact with their documents using an LLM. The process involves setting up a vector database (Chroma) to store document embeddings and connecting it to the LLM for enhanced fact-based responses. The tutorial covers the ingestion of documents, chunking, embedding, and linking the LLM with the knowledge base for retrieval.
- **Link**: [Click here](https://cismography.medium.com/building-a-knowledge-base-for-custom-llms-using-langchain-chroma-and-gpt4all-950906ae496d)

- **Title**: Fine-Tuning LLMs: LoRA or Full-Parameter? An in-depth Analysis with Llama 2
- **Description**: This blog post by Artur Niederfahrenhorst, Kourosh Hakhamaneshi, and Rehaan Ahmad provides an in-depth comparison of full-parameter fine-tuning and LoRA (Low-Rank Adaptation of Large Language Models) for Llama 2 models. The authors demonstrate that while LoRA can achieve comparable performance to full-parameter fine-tuning in specific tasks, it presents a trade-off between model quality and serving efficiency, with the advantage of reduced hardware requirements and training costs.
- **Link**: [Click here](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)

- **Title**: Building a Scalable Pipeline for Large Language Models and RAG : an Overview
- **Description**: The article by Anthony Alcaraz titled "Building a Scalable Pipeline for Large Language Models and RAG: an Overview" discusses the potential of Large Language Models (LLMs) and the importance of Retrieval Augmented Generation (RAG) in enhancing the capabilities of LLMs. The article outlines the steps to construct a scalable pipeline for LLMs and RAG.
- **Link**: [Click here](https://ai.plainenglish.io/building-a-scalable-pipeline-for-large-language-models-and-rag-an-overview-7cb93a03f657)

- **Title**: Build AI search into your applications
- **Description**: The Elasticsearch Relevance Engine™ (ESRE) is designed to power artificial intelligence-based search applications. Use ESRE to apply semantic search with superior relevance out of the box (without domain adaptation), integrate with external large language models (LLMs), implement hybrid search, and use third-party or your own transformer models.
- **Link**: [Click here](https://www.elastic.co/elasticsearch/elasticsearch-relevance-engine)

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.

