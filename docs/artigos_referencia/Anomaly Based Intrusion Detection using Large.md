## Anomaly Based Intrusion Detection using Large Language Models

∗† ∗ ∗

Zineb Maasaoui , Mheni Merzouki , Abdella Battou , Ahmed lbath Smart Connected Systems Division, National Institute of Standards and Technology, MD, USA † LIG/MRIM, CNRS, Grenoble Alpes University, Grenoble, FRANCE

Email: { zineb.maasaoui, mheni.merzouki, abdella.battou } { zineb.maasaoui,ahmed.lbath } @univ-grenoble-alpes.fr

Abstract -In the context of modern networks where cyberattacks are increasingly complex and frequent, traditional Intrusion Detection Systems (IDS) often struggle to manage the vast volume of data and fail to detect novel attacks. Leveraging Artificial Intelligence, specifically Natural Language Processing with transformer architectures, offers a promising solution. This study applies the Bidirectional Encoder Representations from Transformers (BERT) model, enhanced by a Byte-level Byte-pair tokenizer (BBPE), to effectively identify network-based attacks within IoT systems. Experiments on three datasets-UNSWNB15, TON-IoT, and Edge-IIoT-show that our approach substantially outperforms traditional methods in multi-class classification tasks. Notably, we achieved near-perfect classification accuracy on the Edge-IIoT dataset, with significant improvements in F1 scores and reduction in validation losses across all datasets, demonstrating the efficacy of pre-trained Large Language Models (LLMs) in network security.

Index Terms -Network intrusion, Network security, Natural language processing, Large Language Model, BERT, BBPE, AI, IoT

## I. INTRODUCTION

The rapid proliferation of connected devices, especially in Internet of Things (IoT) networks, has significantly increased the complexity and frequency of cyber-attacks, often financially motivated [1]. Traditional security systems such as Intrusion Detection Systems (IDS) and firewalls are increasingly inadequate, struggling to manage vast quantities of network data and to detect novel, sophisticated attacks [2]. In response, the integration of Artificial Intelligence (AI) into network security marks a transformative shift, enhancing threat detection and reducing false positives through advanced machine learning techniques [3], [4].

Recent advancements in Natural Language Processing (NLP) have led to the adoption of transformative architectures, such as the Transformer model introduced by Vaswani et al. [5], which are highly effective at processing unstructured text data. Large Language Models (LLMs) like the Bidirectional Encoder Representations from Transformers (BERT) [6] have demonstrated their utility beyond traditional applications, extending into cybersecurity where they are employed to analyze and interpret network traffic, converting it into structured formats suitable for in-depth analysis.

This research explores the effectiveness of LLMs for anomaly-based intrusion detection within IoT networks. While BERT typically employs a WordPiece tokenizer, our study

† @nist.gov, utilizes a Byte-Level Byte-Pair Encoding (BBPE) tokenizer due to its superior suitability for processing network logs, which are non-linguistic and highly structured. Unlike other studies that use BBPE and BERT, our approach uniquely involves training the BBPE tokenizer on multiple datasets to enrich the vocabulary size, thereby improving normalization and performance on real traffic data. We apply this approach across three datasets: UNSW-NB15, TON-IoT, and EdgeIIoT, achieving significant performance improvements over traditional machine learning and deep learning methods in multi-class classification tasks, including near-perfect accuracy on the Edge-IIoT dataset.

Our contributions are as follows:

- Evaluating the efficacy of Large Language Models (LLMs) in detecting network threats.
- Preprocessing raw network logs to enhance their suitability for analysis using the Byte-Level Byte-Pair Encoding (BBPE) technique.
- Conducting a multi-class classification of network threats across the UNSW-NB15, TON-IoT, and Edge-IIoT datasets, leveraging a pre-trained Bidirectional Encoder Representations from Transformers (BERT) model.
- Comparing the performance of a Byte-Level Byte-Pair Encoding (BBPE) encoder trained on individual datasets versus a combined dataset approach.

To our knowledge, this is the first study to utilize three diverse datasets to train a tokenizer from scratch and detect anomalies in network traffic using Large Language Models (LLMs). The structure of this paper is as follows: Section 2 reviews related work; Section 3 describes the datasets and methods, including a detailed discussion of the LLM architecture, specifically the BERT model and BBPE encoder; Section 4 outlines our proposed approach and data preprocessing steps; Section 5 presents our results and discusses their broader implications; and finally, Section 6 concludes the study.

## II. RELATED WORK

The analysis of network data flow is essential for effective network management, analysis, and threat detection, encompassing various data forms such as logs and pcap files. The initial step in network analysis involves the meticulous collection and examination of network traffic. This task is challenging due to the large volume of data flow. Much

research focuses on the design and implementation of robust platforms for data management, collection, and analysis [2], [7]. Although it may appear straightforward, the quality and comprehensiveness of the dataset are pivotal to the efficacy of Intrusion Detection Systems (IDS). This is true for both traditional signature-based IDS and those utilizing advanced machine learning techniques. High-quality datasets are crucial for developing robust detection models and for training these models to effectively recognize and respond to an evolving array of cyber threats. However, many available network datasets are outdated or lack data on new types of attacks [8]. In this study, we consider three popular datasets: ToN-IoT [9], Edge-IIoTset [10], and UNSW-NB15 [11].

Since the introduction of machine learning techniques in the late 1980s, network security systems, particularly those focusing on anomaly and intrusion detection, have undergone significant evolution. Numerous techniques and algorithms, ranging from basic machine learning to advanced deep learning, have been deployed to enhance these systems. Table I chronicles this progression, highlighting the shift from simple linear models to the complex architectures of transformer models and the integration of pre-trained Large Language Models (LLMs). This section delves into the current state of intrusion detection systems, examining the applications of LLMs within network security. We also focus on their capabilities in advanced text analysis and interpretation, which are crucial for identifying and mitigating emerging cyber threats.

The application of deep learning for intrusion detection in IoT networks is a prominent area of research in academia. A significant study in this field introduced a hybrid CNN-LSTM architecture [18], which effectively addresses large-scale network traffic and zero-day attacks. Validated on 92,209 samples from the ToN-IoT dataset, the model demonstrated high efficiency, achieving over 99% accuracy and 98% precision with a false positive rate of 0.0032. The model benefits from the balanced ToN-IoT dataset, which includes diverse IoT data sources and real-time packet captures, allowing for accurate real-time attack classification during the validation phase.

Despite the effectiveness of CNN-LSTM models, the increasing complexity and volume of network traffic highlight the need for more advanced techniques to detect anomalies. Traditional LSTM-based methods, while powerful, often struggle with the nuanced understanding required for unstructured raw log messages. This has paved the way for leveraging large language models (LLMs) for network intrusion detection. In this context, the authors of BERT-Log [19] introduced a novel approach by utilizing the BERT pre-training language model to automate anomaly detection in log data. The BERT-Log method encompasses an event template extractor, a log semantic encoder, and a log anomaly classifier, significantly outperforming traditional methods such as LSTM, Bi-LSTM, and Word2Vec in capturing semantic information from raw logs. This approach achieved an F1-score of 99.3% on the HDFS dataset. However, the training time of the model was substantial, highlighting an area for future improvement.

Building on this perspective, the NeuralLog approach [20] addresses common issues such as log parsing errors and the inability of traditional methods to handle out-of-vocabulary (OOV) words. NeuralLog eliminates the need for log parsing by directly utilizing raw log messages, thus preserving complete information. It integrates a BERT encoder [6] and a Transformer-based classification model [5] to extract and interpret the semantic meaning of logs. This method not only enhances anomaly detection by capturing contextual nuances but also demonstrates superior performance, achieving F1-scores above 0.95 across four public datasets. The study further suggests potential enhancements, including the integration of more log-specific information to improve detection accuracy.

Building on the progress made by NeuralLog, LogBERT [21] presents a self-supervised framework for detecting log anomalies that leverages raw log data, eliminating the need for traditional parsing. This method introduces two key training tasks: masked log key prediction and reducing hypersphere volume. These tasks enable LogBERT to effectively detect irregularities in log sequences, with performance surpassing other approaches such as DeepLog, particularly when both tasks are employed simultaneously. In evaluations on the HDFS dataset, it achieved an F1 score of 82.32. The dual-task approach proves especially efficient in processing shorter log sequences, emphasizing the Transformer encoder's capacity to discern crucial log patterns, thus setting a high benchmark for network security anomaly detection.

A recent study introduces SecurityBERT [22], an innovative approach that leverages the BERT model for cyber threat detection in IoT networks. This architecture integrates a novel Privacy-Preserving Fixed-Length Encoding (PPFLE) and a Byte-level Byte-Pair Encoder (BBPE) Tokenizer, effectively representing network traffic data for anomaly detection. Tested on the Edge-IIoTset cybersecurity dataset, SecurityBERT achieved an impressive 98.2% accuracy in detecting fourteen distinct types of attacks, surpassing hybrid models such as GAN-Transformer architectures and CNN-LSTM systems. With its rapid inference time and compact model size, SecurityBERT is ideally suited for real-world traffic analysis, offering a robust solution for modern cybersecurity challenges.

Another area of research explores the retrieval-augmented generation technique for system log anomaly detection. The study cited in [23] introduces the Retrieval Augmented Large Language Model (RAGLog). This model uses a vector database in a Question and Answer configuration to handle anomalies. It effectively manages the diversity and volume of

TABLE I PROGRESSION OF MACHINE LEARNING TECHNIQUES AND THEIR APPLICATION IN NETWORK AND SECURITY

| Era         | Key Papers/Works                                                                                                                                                                                                                           | Key Mod- els/Techniques              | Description                                                                                                                                                 | Application in Network and Security                                                           |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 1980s-1990s | • 'Induction of decision trees' by J. R. Quinlan (1986) [12] • 'A training algorithm for opti- mal margin classifiers' by Corinna Cortes and Vladimir Vapnik (1995) [13]                                                                   | • Decision Trees • SVMs              | • Introduction of more com- plex, non-linear models ca- pable of capturing intricate patterns in data.                                                      | • Improved intrusion detec- tion systems.                                                     |
| 2000s       | • 'A decision-theoretic generalization of on-line learning and an applica- tion to boosting' by Yoav Freund and Robert Schapire (1997) [14] • 'Gradient-based learning applied to document recognition' by Yann Le- Cun et al. (1998) [15] | • Ensemble Methods • Neural Networks | • Introduction of ensemble methods and early neural networks, enhancing predic- tion accuracy and stability.                                                | • Advanced malware detec- tion and network behavior analysis.                                 |
| 2010s       | • 'ImageNet Classification with Deep Convolutional Neural Networks' by Alex Krizhevsky et al. (2012) [16]                                                                                                                                  | • Deep Learning                      | • Revolution in ML capa- bilities, particularly through deep neural networks for complex pattern recogni- tion.                                             | • Deep learning in phish- ing detection, fraud analy- sis, and traffic classification.        |
| Late 2010s  | • 'Attention is All You Need' by Ashish Vaswani et al. (2017) [5]                                                                                                                                                                          | • Transformer Mod- els               | • Introduction of transform- ers, focusing on sequence processing with remarkable efficiency and scalability.                                               | • Enhancements in threat in- telligence and SIEM sys- tems using NLP techniques.              |
| 2020s       | • 'BERT: Pre-training of Deep Bidi- rectional Transformers for Language Understanding' by Jacob Devlin et al. (2018) [6] • 'Language Models are Few-Shot Learners' by Tom B. Brown et al. (2020) [17]                                      | • Large Language Models (LLMs)       | • Development of pre-trained models that can be fine- tuned for specific tasks, im- proving upon earlier mod- els' limitations in handling contextual data. | • Advanced threat detection, automated security responses, and real-time security assistance. |

logs across various systems. The approach also compensates for the scarcity of anomalous log entries needed for traditional model training. RAGLog achieved good F1 scores on both the BGL and Thunderbird datasets [24] using a zero-shot method that depends only on normal log entries for analysis. However, the model faces challenges with high resource consumption and slow execution when processing large data volumes.

Additionally, LogPrompt introduces a novel interpretable log analysis using LLMs, significantly reducing the need for in-domain data. Outlined in [25], this method uses advanced prompt strategies, boosting LLM performance by up to 380.7% and surpassing traditional methods by 55.9% across nine datasets. Despite its success, challenges remain, such as the LLM's lack of domain knowledge leading to generic interpretations and difficulty in processing logs with non-NLP patterns like codes and addresses. Feedback from practitioners highlights the potential for real-world applications and underscores the need for enhancing the model's domain-specific understanding through domain adaptation technologies.

This body of work underscores a dynamic interplay between computational needs and communication efficiency in network security, setting the stage for further innovation in LLM applications within cybersecurity.

## III. MATERIALS AND METHODS

The system architecture we propose for detecting network threats integrates the Byte-Pair Encoding (BBPE) encoder with the pre-trained BERT model from Huggingface library. Initially, network log data is transformed into a contextual representation. The BBPE encoder, trained from scratch, with 3 different vocabulary sizes, converts log tokens into vectors. For multi-class classification of various network threats, we utilize the BERT model. We employ the UNSWNB15, ToN-IoT, and Edge-IIoTset datasets as inputs and evaluate system performance using accuracy and F1-score metrics. This section introduces the datasets, describes the BBPE encoder and the BERT architecture, elaborates on the evaluation metrics, and provides a detailed overview of the overall system architecture.

## A. Datasets

This study utilizes three datasets: UNSW-NB15, ToN-IoT, and Edge-IIoTset. The UNSW-NB15 dataset, created by the Cyber Range Lab of UNSW Canberra, includes 100 GB of raw network traffic and captures nine attack types such as DoS, Fuzzers, and Exploits. It consists of 2,540,044 records with 49 features, divided into training and testing sets. The ToN-IoT dataset, developed for Industry 4.0 IoT and IIoT environments, contains telemetry data from IoT sensors, operating system logs, and network traffic. It includes a diverse range of attacks like DoS, DDoS, and ransomware, collected from a realistic large-scale network at UNSW Canberra's Cyber Range and IoT Labs. This dataset supports the evaluation of AI-based cybersecurity applications. The Edge-IIoTset dataset is generated from a purpose-built IoT/IIoT testbed and features data from over 10 types of IoT devices. It identifies and analyzes fourteen types of attacks, categorized into five threat categories: DoS/DDoS, information gathering, man-in-the-middle, injection, and malware attacks. This dataset offers 61 highly correlated features derived from alerts, system resources, logs, and network traffic, making it valuable for intrusion detection research. These datasets enable a comprehensive evaluation of the system's performance using accuracy and F1-score metrics. The attack categories and record distributions across these datasets are summarized in Table II. By combining the datasets, the study benefits from a diverse representation of attacks, enhancing the robustness of the trained models for network intrusion detection.'

TABLE II DATASETS STATISTICS.

| Attack Category   | ToN-IoT   | UNSW-NB15   | Edge-IIoT   | Total     |
|-------------------|-----------|-------------|-------------|-----------|
| Normal            | 50,000    | 93,000      | 1,615,643   | 1,758,643 |
| XSS               | 20,000    | -           | 15,915      | 35,915    |
| Scanning          | 20,000    | -           | 22,564      | 42,564    |
| Ransomware        | 20,000    | -           | 10,925      | 30,925    |
| Password          | 20,000    | -           | 50,153      | 70,153    |
| Injection         | 20,000    | -           | 51,203      | 71,203    |
| DoS               | 20,000    | 16,353      | 50,062      | 86,415    |
| DDoS              | 20,000    | -           | 171,630     | 191,630   |
| Backdoor          | 20,000    | 2,329       | 24,862      | 47,191    |
| MiTM              | 1,043     | -           | 1,214       | 2,257     |
| Generic           | -         | 58,871      | -           | 58,871    |
| Exploits          | -         | 44,525      | -           | 44,525    |
| Fuzzers           | -         | 24,246      | -           | 24,246    |
| Reconnaissance    | -         | 13,987      | -           | 13,987    |
| Analysis          | -         | 2,677       | -           | 2,677     |
| Shellcode         | -         | 1,511       | -           | 1,511     |
| Worms             | -         | 174         | -           | 174       |
| Vul Scanner       | -         | -           | 50,110      | 50,110    |
| DDoS ICMP         | -         | -           | 116,436     | 116,436   |
| DDoS HTTP         | -         | -           | 49,911      | 49,911    |
| Uploading         | -         | -           | 37,634      | 37,634    |
| Port Scanning     | -         | -           | 22,564      | 22,564    |
| Fingerprinting    | -         | -           | 1,001       | 1,001     |
| Attack types      | 9         | 9           | 14          | 22        |

## B. BBPE Encoder

In this study, we utilize the Byte-level Byte-Pair Encoding (BBPE) tokenizer, an adaptation of Byte-Pair Encoding (BPE)

specifically optimized for transformer-based models. BBPE iteratively merges the most frequent pairs of bytes from a corpus to form a subword-based vocabulary, adept at managing outof-vocabulary terms and morphologically complex languages.

For network logs classification, BBPE proves particularly beneficial. It efficiently reduces vocabulary size, which decreases storage needs and accelerates model training. Its resilience against the intrinsic variability and noise in network log data-such as typos, abbreviations, and diverse terminologies-enhances classification accuracy and interpretation. Moreover, BBPE's compatibility with transformer architectures facilitates superior representation learning, significantly boosting performance in network logs classification tasks.

## C. Comprehensive Overview of BERT

BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art machine learning model for natural language processing tasks. Developed by Google [6], BERT is predicated on the Transformer architecture, which diverges from prior text analysis models by processing words in parallel and in a bidirectional context. This model has revolutionized the way machines understand human language.

- 1) Transformer Architecture : At the core of BERT is the Transformer, which employs a multi-head self-attention mechanism. In this mechanism, each word in a text sequence is projected into three vectors:
- Queries Q
- Keys K
- Values V

These vectors are used to compute attention scores using the equation:

<!-- formula-not-decoded -->

where d k is the dimension of the key vectors. This computation signifies the importance of each word relative to others in the sequence, thus capturing contextual relationships in both directions.

- 2) Multi-Head Attention : BERT employs multiple attention 'heads' to explore various relationships in the text independently. Each head computes its own attention scores, and the outputs are concatenated and linearly transformed into the expected dimension.
- 3) Bidirectional Context : Unlike traditional models that read text sequentially, BERT analyzes text in a way that integrates insights from both the left and right contexts of each word. This comprehensive view allows for a deeper understanding of the text, which is crucial for complex language tasks.

## D. Evaluation metrics

To thoroughly evaluate the performance of our model in classifying network logs, we utilize a set of robust metrics,

each offering insights into the model's accuracy, precision, and generalization capabilities across various types of network threats. These metrics are essential for validating the effectiveness of our model and identifying areas for improvement.

## Validation Loss

Validation loss quantifies the model's error on the validation dataset, providing a direct measure of performance:

<!-- formula-not-decoded -->

where y is the vector of true values, and ˆ y is the vector of predicted values by the model. Lower validation loss indicates better generalization.

## Weighted F1 Score

The F1 score is the harmonic mean of precision and recall, particularly important in the presence of class imbalance. It is calculated as:

<!-- formula-not-decoded -->

where Precision and Recall are defined as:

<!-- formula-not-decoded -->

Weighted F1 considers each class's support:

<!-- formula-not-decoded -->

where n j is the number of samples in class j , N is the total number of samples, and F 1 j is the F1 score of class j .

## Accuracy

Overall accuracy is defined as the proportion of true results (both true positives and true negatives) among the total number of cases examined:

<!-- formula-not-decoded -->

where 1 is the indicator function.

## Class-specific Accuracy

Accuracy for each class is critical for understanding model performance on individual threat types, calculated for each class as:

<!-- formula-not-decoded -->

## IV. PROPOSED APPROACH

In this study, we address the challenge of adapting network logs for classification using a BERT model by first transforming these logs into a structured textual format conducive for BBPE and subsequent BERT processing. Our preprocessing approach converts the inherently non-linguistic network logs into a format resembling natural language to optimize compatibility with BERT's architecture.

## A. Data Preprocessing

- 1) Initial Data Cleaning: The raw network logs were initially unsuitable for BBPE tokenizer processing. We enhanced data compatibility by performing several cleaning operations: null values were removed, unnecessary features were eliminated, and normalization processes were applied. The refined data was then stored in CSV format.
- 2) Transformation to Contextual Text Representation: We transformed the network logs into a structured textual format to enable processing with natural language techniques. This transformation involved annotating each data element in a log entry with its corresponding feature name, separating feature-value pairs with spaces. This method ensured the preservation of the integrity and completeness of the original data. Simultaneously, we consolidated multi-attribute logs into a single 'text' attribute for each entry in the standardized dataframe. The matrix representation of logs, denoted by M , facilitated this transformation:

<!-- formula-not-decoded -->

This approach, illustrated in Figure 1, not only preserves detailed log information but also prepares the data for subsequent processing with BBPE and BERT.

Fig. 1. Description of the contextual transformation

<!-- image -->

- 3) Standardization Across Datasets: This preprocessing methodology was consistently applied to three datasets-UNSW-NB15, TON-IoT, and Edge-IoT-each with distinct attributes. By using a uniform dataframe structure labeled with features such as text, label, and category, we ensured that variations in dataset attributes did not compromise the effectiveness of our study.
- 4) Tokenizer Configuration and Training: Following the transformation, the resulting contextual text was used as input for the BBPE tokenizer. The tokenizer was configured with three different vocabulary sizes: 5000, 10.000, and 20.000, incorporating specific tokens. The settings also specified a minimum frequency threshold of 2, optimizing the tokenizer's setup.

Consistent with [22], we trained our BBPE encoder from scratch in four different scenarios. The uniform textual rep-

resentation allowed for the concatenation of all three datasets into a single format, enhancing the richness of the dataset fed into our deep learning model and leveraging the overlapping categories among the datasets.

We explored three scenarios to evaluate the effectiveness of the tokenizer training: the first involved training the tokenizer jointly on all datasets, while the remaining scenarios involved training it separately on each dataset. For each scenario, we varied the vocabulary size between 5000, 10.000, and 20.000, allowing us to assess the impact of vocabulary size on the tokenizer's performance. This comparative analysis aimed to determine whether a unified or individual dataset training approach, coupled with varying vocab sizes, yields superior results. The outcomes of these scenarios are detailed in the subsequent section.

## B. Fine-Tuning the Pre-trained BERT Model

We fine-tuned the pre-trained bert-base-uncased model for the multi-class classification of network logs. The model's output layer was configured to predict the number of classes as specified in our label\_dict , ensuring precise handling of diverse classifications.

For optimization, the AdamW optimizer was employed with a learning rate of 1 × 10 -5 and an epsilon of 1 × 10 -8 . This setup promotes stable and efficient training while minimizing the risk of overfitting.

Prior to training, we conducted extensive preprocessing to align the network logs with BERT's tokenization standards. We first transformed multiple columns into a coherent textual representation. This representation was then processed by the BBPE tokenizer to generate consistent embeddings, specifically input IDs and attention masks. These embeddings are crucial for enabling BERT to process the complex features within the logs accurately. During training, batches comprising input IDs, attention masks, and labels were fed into the model, allowing it to adapt to the nuances of our specific dataset and significantly enhance its classification performance across various network log categories.

## V. RESULTS AND DISCUSSION

We assessed the performance of our fine-tuned BERT model on three distinct datasets-UNSW-NB15, TON-IoT, and EdgeIIoT-focusing on multi-class classification tasks. Our primary evaluation metrics were validation loss and weighted F1 scores, offering a comprehensive insight into the model's performance across various training epochs.

## A. Dataset-Specific Performance

- TON-IoT : Exhibited high accuracy, with F1 scores progressing from 0.9899 to 0.9938 across four epochs. The lowest recorded validation loss was 0.0191, reflecting effective model learning and generalization capabilities. A notable observation from the results is the impact of vocabulary size on classification performance. As shown in Figure 3, while all vocabulary sizes led to strong performance, the models trained with larger vocabulary
- size 20000 and on all datasets exhibited marginally better F1 scores compared to the 5000-vocabulary model, particularly in later epochs. This suggests that increasing the vocabulary size provides the model with richer token representations, which is particularly advantageous for complex multi-class classification tasks, such as those in the TON-IoT dataset.
- UNSW-NB15 : As shown in Figure 2, the tokenizer trained on all datasets with a vocabulary size of 20,000 demonstrates significant improvements in both training and validation metrics over the first five epochs. The F1 score steadily increases, peaking at 0.8554 by epoch 5, while the validation loss decreases to 0.3809, indicating optimal model performance. Both training and validation F1 scores show clear progression, reflecting the model's ability to generalize effectively with a larger and more diverse vocabulary. The decrease in validation loss suggests that the model learns meaningful patterns and avoids overfitting during these early epochs. This highlights the importance of using a larger vocabulary size to capture complex network traffic patterns.
- Edge-IIoT : The model achieved near-perfect accuracy across all classes, with an F1 score of 0.99999 and a validation loss as low as 0.0000441 by the third epoch. Even for minority classes like Ransomware (97% accuracy) and MITM (99% accuracy), the model maintained strong performance, demonstrating its robustness in handling class imbalance. Despite potential concerns of overfitting due to the high accuracy and low validation loss, future work will test the generalizability of the finetuned model by applying it to other datasets to ensure its broader applicability. This approach will help confirm whether the model overfits to the EDGE-IIoT dataset or generalizes effectively to unseen data from different network environments.

Fig. 2. Training and Validation F1 Scores and Loss Across Epochs on the UNSW-NB15 Dataset

<!-- image -->

## B. Analysis of Multi-Class Accuracies

This subsection compares class-wise accuracies across the TON-IoT, UNSW-NB15, and Edge-IIoT datasets, using tok-

Fig. 3. F1 Score Across Epochs for Different Vocabulary Sizes on the TONIoT Dataset

<!-- image -->

enizers trained individually (Sgl) and in combination (Cmb). For the UNSW-NB15 dataset, the combined tokenizer was trained with a vocabulary size of 20,000 and for 5 epochs, compared to the smaller vocabulary size of 5,000 used when the tokenizer was trained solely on UNSW-NB15. This approach shows improved performance across key classes, such as Normal, Backdoor, and DDoS, indicating that a larger and more diverse vocabulary enhances detection in the UNSWNB15 dataset.

However, despite these improvements, classes like DoS and Backdoor in the UNSW-NB15 dataset still perform poorly, even with the combined tokenizer, suggesting ongoing datasetspecific challenges. In contrast, well-represented classes such as Ransomware and Injection perform consistently well across all datasets, benefiting from the diverse vocabulary and combined training approach.

## VI. CONCLUSION AND FUTURE WORK

This research has demonstrated the vital importance of integrating diverse network and IoT datasets to enhance data quality and quantity for anomaly detection. Through effective data transformation and the application of Large Language Models (LLMs), we have successfully navigated the challenges posed by disparate dataset features. Our findings particularly highlight the benefits of this integration, with improved model performance across various attack categories.

In the specific case of the UNSW-NB15 dataset, which initially had low occurrence rates in certain attack categories, extending the training to over 40 epochs significantly enhanced the accuracy for 'Backdoor' and 'DoS' attacks while reducing the overall loss. These results indicate the potential of prolonged training periods in optimizing the detection capabilities of LLMs.

Looking ahead, our future work will focus on further exploring this promising area of research. We aim to refine our approach by investigating strategies to reduce computing costs associated with extensive training. Additionally, we will experiment with increasing the number of epochs to ascertain the optimal balance between computational efficiency and

TABLE III DETAILED MULTI-CLASS ACCURACIES ACROSS DATASETS. Note: 'Sgl' refers to the tokenizer trained on the dataset alone, while 'Cmb' refers to the tokenizer trained on all three datasets combined.

| Class                 | TON-IoT   | TON-IoT   | UNSW-NB15   | UNSW-NB15   | Edge-IIoT   | Edge-IIoT   |
|-----------------------|-----------|-----------|-------------|-------------|-------------|-------------|
| Class                 | Sgl       | Cmb       | Sgl         | Cmb         | Sgl         | Cmb         |
| Normal                | 0.99      | 1.00      | 0.96        | 0.98        | 0.99        | 1.00        |
| Backdoor              | 0.99      | 1.00      | 0.00        | 0.17        | 0.97        | 1.00        |
| DDoS                  | 0.92      | 0.99      | -           | -           | 1.00        | 1.00        |
| DoS                   | 0.95      | 0.99      | 0.02        | 0.28        | -           | -           |
| MITM                  | 0.97      | 0.99      | -           | -           | 0.99        | 1.00        |
| Ransomware            | 0.99      | 1.00      | -           | -           | 0.92        | 1.00        |
| Injection             | 0.98      | 0.99      | -           | -           | 0.98        | 1.00        |
| Scanning              | 0.99      | 0.99      | -           | -           | 0.98        | 1.00        |
| XSS                   | 0.87      | 1.00      | -           | -           | 0.99        | 1.00        |
| Exploits              | -         | -         | 0.93        | 0.93        | -           | -           |
| Reconnaissance        | -         | -         | 0.75        | 0.80        | -           | -           |
| Fuzzers               | -         | -         | 0.65        | 0.75        | -           | -           |
| Analysis              | -         | -         | 0.05        | 0.17        | -           | -           |
| Shellcode             | -         | -         | 0.68        | 0.83        | -           | -           |
| Password              | 0.99      | 1.00      | -           | -           | 1.00        | 1.00        |
| Port Scanning         | -         | -         | -           | -           | 0.96        | 1.00        |
| SQL Injection         | -         | -         | -           | -           | 0.99        | 1.00        |
| Vulnerability Scanner | -         | -         | -           | -           | 0.99        | 1.00        |
| DDoS-TCP              | -         | -         | -           | -           | 1.00        | 1.00        |
| DDoS-HTTP             | -         | -         | -           | -           | 1.00        | 1.00        |
| DDoS-UDP              | -         | -         | -           | -           | 1.00        | 1.00        |
| DDoS-ICMP             | -         | -         | -           | -           | 1.00        | 1.00        |
| Fingerprinting        | -         | -         | -           | -           | 1.00        | 1.00        |
| Uploading             | -         | -         | -           | -           | 1.00        | 1.00        |

model performance. This continued research will contribute to the development of more robust and cost-effective AI-driven security mechanisms for complex network environments.

## VII. DISCLAIMER

Any mention of commercial products or reference to commercial organizations is for information only; it does not imply recommendation or endorsement by NIST, nor does it imply that the products mentioned are necessarily the best available for the purpose.

## REFERENCES

- [1] Z. K. Goldman and D. McCoy, 'Deterring financially motivated cybercrime,' J. Nat'l Sec. L. &amp; Pol'y , vol. 8, p. 595, 2015.
- [2] Z. Maasaoui, M. Merzouki, A. Bekri, A. Abane, A. Battou, and A. Lbath, 'Design and implementation of an automated network traffic analysis system using elastic stack,' in 2023 20th ACS/IEEE International Conference on Computer Systems and Applications (AICCSA) . IEEE, 2023, pp. 1-6.
- [3] E. Rehman, M. Haseeb-ud Din, A. J. Malik, T. K. Khan, A. A. Abbasi, S. Kadry, M. A. Khan, and S. Rho, 'Intrusion detection based on machine learning in the internet of things, attacks and counter measures,' The Journal of Supercomputing , pp. 1-35, 2022.
- [4] M. Aledhari, R. Razzak, and R. M. Parizi, 'Machine learning for network application security: Empirical evaluation and optimization,' Computers &amp; Electrical Engineering , vol. 91, p. 107052, 2021.
- [5] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, 'Attention is all you need,' Advances in neural information processing systems , vol. 30, 2017.
- [6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, 'Bert: Pre-training of deep bidirectional transformers for language understanding,' arXiv preprint arXiv:1810.04805 , 2018.

- [7] Z. Maasaoui, A. Hathah, H. Bilil, V. S. Mai, A. Battou, and A. Lbath, 'Network security traffic analysis platform-design and validation,' in 2022 IEEE/ACS 19th International Conference on Computer Systems and Applications (AICCSA) . IEEE, 2022, pp. 1-5.
- [8] G. Mohi-ud din, 'Nsl-kdd,' 2018. [Online]. Available: https://dx.doi.org/10.21227/425a-3e55
- [9] N. Moustafa, 'Ton-iot datasets,' 2019. [Online]. Available: https://dx.doi.org/10.21227/fesz-dm97
- [10] M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke, 'Edge-iiotset: A new comprehensive realistic cyber security dataset of iot and iiot applications: Centralized and federated learning,' 2022. [Online]. Available: https://dx.doi.org/10.21227/mbc1-1h68
- [11] I. Tareq, B. M. Elbagoury, S. El-Regaily, and E.-S. M. El-Horbaty, 'Analysis of ton-iot, unw-nb15, and edge-iiot datasets using dl in cybersecurity for iot,' Applied Sciences , vol. 12, no. 19, p. 9572, 2022.
- [12] J. R. Quinlan, 'Induction of decision trees,' Machine learning , vol. 1, pp. 81-106, 1986.
- [13] B. E. Boser, I. M. Guyon, and V. N. Vapnik, 'A training algorithm for optimal margin classifiers,' in Proceedings of the fifth annual workshop on Computational learning theory , 1992, pp. 144-152.
- [14] Y. Freund and R. E. Schapire, 'A decision-theoretic generalization of on-line learning and an application to boosting,' Journal of computer and system sciences , vol. 55, no. 1, pp. 119-139, 1997.
- [15] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, 'Gradient-based learning applied to document recognition,' Proceedings of the IEEE , vol. 86, no. 11, pp. 2278-2324, 1998.
- [16] A. Krizhevsky, I. Sutskever, and G. E. Hinton, 'Imagenet classification with deep convolutional neural networks,' Advances in neural information processing systems , vol. 25, 2012.
- [17] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al. , 'Language models are few-shot learners,' Advances in neural information processing systems , vol. 33, pp. 1877-1901, 2020.
- [18] I. B. Ahmed, F. B. Ktata, and K. B. Kalboussi, 'Dla-abids: Deep learning approach for anomaly based intrusion detection system,' in 2023 20th ACS/IEEE International Conference on Computer Systems and Applications (AICCSA) . IEEE, 2023, pp. 1-8.
- [19] S. Chen and H. Liao, 'Bert-log: Anomaly detection for system logs based on pre-trained language model,' Applied Artificial Intelligence , vol. 36, no. 1, p. 2145642, 2022. [Online]. Available: https://doi.org/10.1080/08839514.2022.2145642
- [20] V.-H. Le and H. Zhang, 'Log-based anomaly detection without log parsing,' in 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE) . IEEE, 2021, pp. 492-504.
- [21] H. Guo, S. Yuan, and X. Wu, 'Logbert: Log anomaly detection via bert,' in 2021 international joint conference on neural networks (IJCNN) . IEEE, 2021, pp. 1-8.
- [22] M. A. Ferrag, M. Ndhlovu, N. Tihanyi, L. C. Cordeiro, M. Debbah, and T. Lestable, 'Revolutionizing cyber threat detection with large language models,' arXiv preprint arXiv:2306.14263 , 2023.
- [23] J. Pan, S. L. Wong, and Y. Yuan, 'Raglog: Log anomaly detection using retrieval augmented generation,' arXiv preprint arXiv:2311.05261 , 2023.
- [24] A. Oliner and J. Stearley, 'What supercomputers say: A study of five system logs,' in 37th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN'07) , 2007, pp. 575-584.
- [25] Y. Liu, S. Tao, W. Meng, J. Wang, W. Ma, Y. Zhao, Y. Chen, H. Yang, Y. Jiang, and X. Chen, 'Logprompt: Prompt engineering towards zeroshot and interpretable log analysis,' arXiv preprint arXiv:2308.07610 , 2023.