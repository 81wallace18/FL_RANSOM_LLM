## Transfer Learning in Pre-Trained Large Language Models for Malware Detection Based on System Calls

Pedro Miguel S´ anchez S´ anchez ∗ 1 , Alberto Huertas Celdr´ an 2 , G´ erˆ ome Bovet 3 , Gregorio Mart´ ınez P´ erez 1

Abstract -In the current cybersecurity landscape, protecting military devices such as communication and battlefield management systems against sophisticated cyber attacks is crucial. Malware exploits vulnerabilities through stealth methods, often evading traditional detection mechanisms such as software signatures. The application of ML/DL in vulnerability detection has been extensively explored in the literature. However, current ML/DL vulnerability detection methods struggle with understanding the context and intent behind complex attacks. Integrating large language models (LLMs) with system call analysis offers a promising approach to enhance malware detection. This work presents a novel framework leveraging LLMs to classify malware based on system call data. The framework uses transfer learning to adapt pre-trained LLMs for malware detection. By retraining LLMs on a dataset of benign and malicious system calls, the models are refined to detect signs of malware activity. Experiments with a dataset of over 1TB of system calls demonstrate that models with larger context sizes, such as BigBird and Longformer, achieve superior accuracy and F1-Score of approximately 0.86. The results highlight the importance of context size in improving detection rates and underscore the trade-offs between computational complexity and performance. This approach shows significant potential for real-time detection in high-stakes environments, offering a robust solution to evolving cyber threats.

Index Terms -Large Language Model, Malware Detection, System Call, Transfer Learning, Cybersecurity

## I. INTRODUCTION

In the contemporary landscape of cybersecurity, particularly within the defense sector, safeguarding military assets represents a critical concern. Military devices, often connected to different networks, are frequent targets of sophisticated cyberattacks [1]. Malware, a predominant threat vector, exploits vulnerabilities through deceptive and often undetected methods such as zero-day vulnerabilities [2]. The detection and neutralization of such threats are paramount, not only to

∗ Corresponding author.

1 Pedro Miguel S´ anchez S´ anchez and Gregorio Mart´ ınez P´ erez are with the Department of Information and Communications Engineering, University of Murcia, 30100 Murcia, Spain (pedromiguel.sanchez@um.es; gregorio@um.es) .

2 Alberto Huertas Celdr´ an are with the Communication Systems Group (CSG) at the Department of Informatics (IfI), University of Zurich UZH, 8050 Z¨ urich, Switzerland (e-mail: huertas@ifi.uzh.ch).

3 G´ erˆ ome Bovet is with the Cyber-Defence Campus within armasuisse Science &amp; Technology, 3602 Thun, Switzerland (gerome.bovet@armasuisse.ch) .

This work has been partially supported by (a) the Swiss Federal Office for Defense Procurement (armasuisse) with the DATRIS and CyberForce projects, and (b) the University of Z¨ urich UZH, and (c) the strategic project CDLTALENTUM from the Spanish National Institute of Cybersecurity (INCIBE) by the Recovery, Transformation, and Resilience Plan, Next Generation EU.

ensure the operational integrity of military systems but also to protect critical security interests.

Incorporating artificial intelligence (AI) into cybersecurity represents a transformative shift in how defenses are conceptualized and deployed. AI technologies, through their ability to process and analyze vast amounts of data at high performance, provide a significant advantage in identifying and responding to cyber threats [3]. Machine Learning and Deep Learning (ML/DL) algorithms, a subset of AI, are particularly adept at learning from and adapting to new information, thereby continuously improving threat detection models [4]. This dynamic capability is crucial in an environment where threat actors continually evolve their methods to bypass conventional security measures.

Recent advancements in artificial intelligence (AI), specifically through the development of large language models (LLMs) for Natural Language Processing (NLP), have introduced promising new methodologies for enhancing cybersecurity measures [5]. Some application scenarios involve spam filtering, smart contract auditing, or code analysis, among others. Among these solutions, transfer learning emerges as a particularly efficacious strategy. This approach leverages the extensive knowledge base of LLMs, originally trained on vast datasets, to adapt them to specialized domains with relatively sparse data, such as malware detection in military systems [6].

Despite the advances in the application of LLMs in the cybersecurity field, some challenges remain open: (i) adapting LLMs to effectively process non-linguistic data such as system calls, which may not inherently fit the natural language processing paradigms these models were originally designed for; (ii) ensuring real-time detection capabilities within operational constraints, as LLMs can be computationally intensive and may not suit the time-sensitive needs of military operations; (iii) handling the trade-offs between context length and detection accuracy, as longer context may improve accuracy but also increases computational demands and latency; (iv) scaling the models to handle diverse and evolving attack vectors without extensive manual updates.

To solve the previous challenges, the main contributions of this work are:

- A framework that utilizes system call traces from the device and an LLM model for the detection and classification of malware samples. The framework leverages pre-trained LLMs to refine their knowledge and add a

classification layer on top of the model.

- The proposed framework is validated using a real dataset containing +1TB system calls collected in an RPi3-based spectrum sensor [7]. Several LLMs are tested, and their performance is compared according to factors such as the attention window. The best results are achieved with BigBird and Longformer models as both of them obtain ≈ 0 . 86 in typical classification metrics such as accuracy or F1-Score
- A discussion analyzing the trade-off between detection speed and performance. It analyzes the processed data and the achieved results, remarking how LLM-based outputs can be later processed to achieve higher performance.

The remainder of this article is organized as follows. Section II reviews the current state-of-the-art in language model-based syscall analysis and LLMs in cybersecurity. Section III describes the design of the proposed solution employing LLMs for malware detection. Section IV validates the solution in a real-world syscall dataset containing malware execution samples. Section V reflects about the achieved results and the advantages and drawbacks of the solution. Finally, Section VI depicts the conclusions obtained and the future work.

## II. RELATED WORK

This section reviews the current state-of-the-art on the application of LLMs, and other NLP methods, in the area of anomaly detection, security, and malware detection using system calls as inputs.

N-gram-based methods for system call processing have been utilized as a baseline in the field of cybersecurity and malware detection [8]. More recently, n-gram techniques have been applied as feature extraction approaches for ML/DL-based methods [9], [10]. However, their primary limitation lies in the lack of contextual understanding, as n-grams consider only fixed-length sequences and cannot capture longer dependencies or complex behaviors that span beyond the defined n-size window. This often results in a higher rate of false positives and negatives, particularly with sophisticated malware that employs evasion techniques or mimics benign behaviors.

Regarding LLMs for syscall processing, the authors of [11] compared LSTM, Transformers, and Longformers (a Transformer variant with lower complexity) with 4-gram statistical approaches for syscall classification. They published a dataset covering the system calls generated by seven behaviors monitored in two million web requests. The neural networkbased solutions achieved better performance, especially for LSTM and Transformer. They remarked on the challenges of real-time execution of complex language models.

Focused on intrusion detection, the authors of [12] proposed an ensemble method based on LSTM models that achieved a 0.928 AUC (Area Under Curve), 0.16 FAR (False Alarm Rate) for 0.90 (Detection Rate), in relevant literature syscall datasets such as AFDA-LD and KDD98 [13]. In [14], authors demonstrated that a Longformer-based approach improves previous BERT-based solution [15] in system log anomaly detection due to its larger context size. Data leveraged in this study contains logs from HFDS and Thunderbird applications and the BGL supercomputer system.

As can be seen in the related work comparison in TABLE I and recent literature reviews confirm [5], LLMs have not been extensively applied in syscall data in the realm of intrusion or malware detection, remaining a field requiring considerable research efforts.

TABLE I: LLMs and NLP Methods in Syscall Processing and/or Cybersecurity

| Work   |   Year | Approach                           | Model                         | Results                                                            |
|--------|--------|------------------------------------|-------------------------------|--------------------------------------------------------------------|
| [11]   |   2023 | Syscall behavior classification    | LSTM, Transformer, Longformer | Improved perfor- mance over 4- gram                                |
| [12]   |   2016 | Develop intrusion detection system | LSTM ensemble                 | AUC 0.928, FAR 0.16                                                |
| [14]   |   2022 | Application log anomaly detection  | Longformer- based             | Larger context size improved BERT and LSTM models                  |
| This   |   2024 | Malware classifi- cation           | Longformer, BigBird           | +0.86 accuracy, precision, recall, F1-score. Context size analysis |

## III. DESIGN

This section details the design of the framework for malware detection and classification based on system calls. Fig. 1 describes the proposed framework and the interaction between its components.

## A. Data Gathering

This module is responsible for capturing system calls generated by applications running on the device. System calls (syscalls), which represent the interface between user applications and the operating system, provide a rich source of information about application behavior. This module employs a syscall monitor at the operating system level, which hooks into the syscall dispatcher to log relevant information such as syscall identifiers, arguments, return values, and timestamps. The collected data is crucial for understanding the actions performed by applications, enabling the identification of potentially malicious behavior patterns. By continuously monitoring and recording syscalls, this module ensures that comprehensive and detailed data is available for subsequent analysis. Once syscalls are captured and logged, they can either be sent to an external server for processing or processed on-device, depending on the device processing capabilities.

## B. Data Preprocessing

The preprocessing module prepares the captured syscall data for analysis by the LLM. It primarily focuses on data tokenization and batching. Data tokenization involves converting raw syscall data into a format that the LLM can process, typically by mapping the text of each syscall and its attributes to unique tokens. These tokens preserve the sequence

Fig. 1: LLM-based malware detection framework

<!-- image -->

and contextual information necessary for accurate analysis. Note that the tokenizer should be configured according to the LLM deployed later in the pipeline. Once tokenized, the data is organized into batches to facilitate efficient processing by the LLM. Batching helps manage memory usage and speeds up the computation by allowing parallel processing of multiple data sequences. This streamlined preprocessing ensures that the data is in an optimal state for subsequent classification by the LLM.

## C. LLM-based Classification

At the core of the framework is the LLM-based classification module. This module performs the critical task of classifying syscalls to determine whether they indicate malware activity. The LLM analyzes the sequences and contexts of incoming syscall data and classifies them as either normal or malicious based on the learned patterns. Pre-trained LLMs are leveraged in this module, adapting their final layers for classification and applying a transfer learning process using the syscall data.

1) Model Training: This submodule is responsible for adapting the LLM to the specific task of syscall classification. The training involves supervised learning using labeled datasets comprising examples of both benign and malicious syscall patterns. During training, the model learns to differentiate between these patterns by adjusting its parameters. The goal is to minimize classification errors and improve the model accuracy in distinguishing between benign and malicious behaviors.

- 2) Model Inference: Once trained, the inference submodule uses the model to perform real-time or batch analysis of incoming syscall data. Based on the learned patterns, it classifies the data as either malicious or benign. This real-time analysis ensures timely detection and response to potential threats.

## D. Decision Making

The Decision Making module processes outputs from the LLM to determine the presence of malware, utilizing probabilities to evaluate the likelihood of malicious activity. Userdefined thresholds adjust sensitivity and specificity, allowing for tailored detection settings. Lower thresholds increase sensitivity but may result in more false positives, while higher thresholds enhance specificity, potentially missing some threats. Beyond threshold-based decisions, the module can aggregate LLM outputs over time or across devices to detect broader patterns. Additionally, the probabilities can train a secondary model, such as a logistic regression or decision tree, to refine classifications. This approach enhances precision by leveraging the LLM scores. The module integrates these techniques to ensure accurate and appropriate responses, such as alerting users, quarantining processes, or blocking malicious activities, thereby enhancing the framework overall reliability and effectiveness.

## IV. VALIDATION

This section describes the experiments performed to validate the proposed framework. This validation focused on the verification of the capabilities of LLMs to detect and classify malware samples, as the data gathering part of the framework has been implemented in the literature previously [10].

## A. MalwSpecSys Dataset

The selected dataset for validation is MalwSpecSys [7]. This dataset models the internal behavior of an IoT spectrum sensor belonging to the ElectroSense platform [16], consisting of a Raspberry Pi 3 with a software-defined radio kit. The behavior was monitored when it was functioning normally and under different malware attacks. Syscalls were collected using the perf trace tool.

This dataset contains +1 TB of information collected during several days. The data consists of raw syscall information with the format [ timestamp , process , PID , syscall ], and is divided into one file every 10 seconds.

Regarding the malware samples deployed during data collection, they have different varieties of common cyberattacks:

- Bashlite [17], a well-known botnet family targeting IoT devices. It is capable of launching distributed denialof-service (DDoS) attacks, executing arbitrary shell commands, and enlisting infected devices into a botnet.
- TheTick [18], a backdoor used to control bots from a remote location through a server utilizing a remote shell and retrieving data from targeted devices
- Bdvl [19], a rootkit with very wide functionality. It ranges from hidden backdoors that allow multiple connection methods to keylogging and stealing passwords and files.
- RansomwarePoC [20], an example of ransomware implementation in Python with full encryption capabilities. Only the C&amp;C functionality is missing.

As the syscalls generated by normal behavior and each attack are known, the problem is addressed as a classification task with five different labels, one for normal behavior and four for the different malware.

For preprocessing, the timestamp , process , and PID are removed from the dataset, maintaining only the syscalls ordered in time and without call parameters. Besides, numerous nanosleep syscalls are removed as they do not provide useful information about the system activities. After processing, each file contains ≈ 23k syscalls. These sequences are the ones that are fed as input in the LLM models. The number of syscalls per sequence varies according to the maximum context length of the models.

## B. LLMs for Transfer-Learning

Once the data is ready, diverse pre-trained LLMs are tested. A final layer with five neurons is added to each model to adapt it to the classification task, having one neuron per class. The base tokenizer of each LLM is employed to preprocess the text data and make it appropriate for the model format.

Then, five training epochs are executed to personalize the LLM to the new task. For implementation purposes, Huggingface's Trasnsformers library is employed [21]. As the optimizer function, AdamW with a 1 e -5 learning rate is employed in all cases. The experiments are executed in a compute node leveraging an AMD EPYC 7742 CPU and an NVIDIA A100 40GB GPU. Although more GPUs were available, only one was used in order to be representative of real-world complete framework deployments involving LLM retraining where exhaustive computing resources are not available.

The selected LLMs for validation are open-source and commonly utilized in transfer learning problems. Besides, the chosen models remain relatively small, avoiding using models with billions of parameters. Each model is characterized by its context size, impacting its ability to handle long sequences of text effectively.

- BERT (Bidirectional Encoder Representations from Transformers) has a context size of 512 tokens. It is widely used for various natural language understanding tasks due to its bidirectional attention mechanism, which allows it to capture context from both directions.
- DistilBERT , a smaller and faster version of BERT, also with a context size of 512 tokens. It maintains 97% of BERT's language understanding capabilities while being more efficient, making it suitable for resource-constrained environments.
- GPT-2 (Generative Pre-trained Transformer 2) supports a context size of 1024 tokens. It is known for its strong
- text generation capabilities, leveraging its autoregressive nature to produce coherent and contextually relevant text.
- BigBird , with a context size of 4096 tokens, extends the Transformer architecture to handle longer sequences efficiently. It combines sparse attention mechanisms to manage large contexts, making it ideal for tasks requiring extensive contextual information.
- Longformer supports context sizes of up to 16384 tokens depending on the implementation. It employs a combination of local and global attention mechanisms to efficiently process long documents, particularly useful for tasks like document classification and summarization. The model tested uses a context size of 4096.
- Mistral Small , designed with an impressive maximum context size of 128,000 tokens, utilizes advanced techniques like sliding window attention, making it ideal for tasks involving extensive context and long-form content generation. The model tested uses a context size of 8192.

Several classification metrics based on the confusion matrix are calculated to measure classification performance. The confusion matrix is a table used to describe the performance of a classification model on a set of data for which the true values are known. It classifies predictions into four categories: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

- Accuracy evaluates the overall correctness of the model and is calculated as the ratio of correctly predicted observations to the total observations.

<!-- formula-not-decoded -->

- Precision or True Positive Rate (TPR) assesses the accuracy of positive predictions made by the model.

<!-- formula-not-decoded -->

- Recall (or sensitivity) measures the model ability to detect positive samples.

<!-- formula-not-decoded -->

- The F1-Score is the harmonic mean of precision and recall, providing a balance between the two by considering both false positives and false negatives.

<!-- formula-not-decoded -->

- Cohen's Kappa statistic adjusts accuracy by accounting for the possibility of agreement occurring by chance.

<!-- formula-not-decoded -->

- Matthews Correlation Coefficient (MCC) is a reliable statistical rate that produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (TP, TN, FP, FN), proportionally.

<!-- formula-not-decoded -->

TABLE II details the results achieved for each one of the tested LLMs. It can be seen how the increase in the

TABLE II: Classification Results of LLMs

| Model      |   Context Size |   Accuracy |   Precision |   Recall |   F1-Score |   Kappa |    MCC |
|------------|----------------|------------|-------------|----------|------------|---------|--------|
| BERT       |            512 |     0.6772 |      0.82   |   0.6596 |     0.6465 |  0.5504 | 0.6024 |
| DistilBERT |            512 |     0.6289 |      0.71   |   0.6181 |     0.593  |  0.4786 | 0.5379 |
| GPT-2      |           1024 |     0.6944 |      0.7986 |   0.6865 |     0.6808 |  0.5792 | 0.6123 |
| BigBird    |           4096 |     0.8667 |      0.8754 |   0.8668 |     0.8688 |  0.8298 | 0.8311 |
| Longformer |           4096 |     0.8616 |      0.8696 |   0.8614 |     0.8621 |  0.8232 | 0.825  |
| Mistral    |           8192 |     0.5817 |      0.6112 |   0.6462 |     0.6242 |  0.4754 | 0.4798 |

context size improves the model performance sequentially until it reaches 4096 tokens with Longformer and BigBird models. After that limit, the Mistral model with 8192 context size performed worse than other models because Normal and TheTick behaviors were misclassified. In contrast, Bashlite, Bdvl, and RansomwarePoC behaviors were reliably classified with +0.96 TPR.

Fig. 2 shows the confusion matrix for the BigBird model, the best-performing model. It illustrates the classification performance across the different malware families and normal behavior. The model achieved a high TPR for normal behavior at 0.8842 but misclassified 0.205 as Bdvl and 0.946 as TheTick. For Bashlite, the accuracy was 0.8900, with minimal misclassifications. Bdvl was correctly identified with a TPR of 0.9921. RansomwarePoC showed a TPR of 0.9724 with minor misclassifications. TheTick had a lower TPR of 0.5952, with 0.3482 misclassified as normal behavior. The results highlight the effectiveness of BigBird in identifying specific threats and the need for further refinement in detecting TheTick.

Fig. 2: BigBird Confusion Matrix

<!-- image -->

## V. DISCUSSION

The results of the validation underscore the importance of context size in enhancing the performance of LLMs for malware classification tasks. It is observed that models with larger context sizes, such as BigBird and Longformer, achieved higher accuracy and better classification metrics compared to those with smaller context sizes, like BERT and DistilBERT. Specifically, BigBird, with a context size of 4096 tokens, demonstrated superior performance in accurately classifying the syscall patterns of various malware types, as evidenced by its high TPR across multiple classes. The main reason for the performance improvement with the context size increase is the large number of syscalls being generated per second. The dataset contains roughly 23,000 syscalls every 10 seconds, which is a rate of +2,000 syscalls per second. Therefore, models with context sizes of 512 or 1024 are not able to process the syscalls generated every second. During that short period, the malware samples might not perform any operations, even if they are active in the device.

However, the results also reveal some trade-offs between context size and performance. While increased context size generally improves the model ability to capture and utilize extended sequences of syscalls, it also introduces challenges. For instance, the Mistral model, despite its capacity to handle up to 128,000 tokens, showed reduced performance at a context size of 8192 tokens. This drop in performance can be attributed to the increased complexity and potential overfitting when dealing with excessively large contexts without sufficient data diversity or volume to support such detailed analysis.

The misclassification of TheTick as normal behavior highlights a critical area for further refinement. This suggests that while the model can effectively utilize larger contexts to improve detection rates for certain malware types, it may still struggle with specific patterns or classes that require more nuanced differentiation. This finding points to the need for balanced context sizes that maximize information utility without overwhelming the model discriminative capabilities.

Another significant aspect to consider is how LLM predictions can be aggregated to enhance detection accuracy. Advanced detection systems can benefit from ensemble methods, where predictions from multiple models with varying context sizes are combined to form a consensus decision. This approach leverages the strengths of each model, potentially offsetting their individual weaknesses. For instance, predictions from BERT and DistilBERT, which are efficient and effective for shorter contexts, can be combined with those from BigBird and Longformer for a more comprehensive analysis. Weighted averaging, voting mechanisms, or more sophisticated ensemble techniques like stacking can be employed to aggregate predictions, thereby improving overall detection robustness.

Works in the literature dealing with similar scenarios, such as [22], have achieved higher detection rates in the malware evaluated in this work using kernel events as data source. Concretely, using ten-second windows, a 0.94 average F1score was achieved. However, note that the 4096 context size in the best-performing LLMs of this work (Longformer and BigBird) allows the processing of around one second of syscall data per evaluation. Therefore, the aggregation of the predictions would be necessary to achieve higher performance with similar time windows.

Integrating LLM predictions with additional context-specific features, such as temporal patterns of syscalls or correlation with network activities, could further enhance the detection framework. This multi-faceted approach allows for the consideration of not just static syscall patterns but also their dynamic behavior over time, providing a richer context for identifying sophisticated malware that employs evasion techniques.

## VI. CONCLUSIONS AND FUTURE WORK

This work proposes a malware detection framework based on LLM transfer learning. It leverages LLMs ability to process and analyze sequences of syscalls. The data preprocessing module ensures the syscalls are tokenized and batched for efficient processing. The core of the framework is the LLM-based classification module, which analyzes the syscall sequences and classifies them as benign or malicious. The decisionmaking module processes the classification results, applying thresholds to determine the presence of malware.

During validation, pre-trained LLMs, including BERT, DistilBERT, GPT-2, BigBird, Longformer, and Mistral, were adapted with an additional classification layer for syscall analysis. Using a dataset of over 1TB of system calls from a Raspberry Pi 3-based spectrum sensor, that captured both normal and malicious activities for training and validation.

The results showed that models with larger context sizes, such as BigBird and Longformer, achieved better classification metrics compared to those with smaller context sizes. Specifically, BigBird and Longformer, with a context size of 4096 tokens, demonstrated superior performance, achieving accuracy and F1-Score of approximately 0.86. However, the Mistral model, despite its ability to handle up to 128,000 tokens, performed worse at a context size of 8192 tokens. This reduced performance is attributed to the complexity and potential overfitting associated with excessively large contexts.

Future work considers applying model quantization techniques to deploy the LLMs in the devices themselves, avoiding the need for external processing. Another future direction is adapting the LLMs for anomaly detection using only benign data, as this could potentially allow the detection of zero-day attacks not seen previously. Additional contextspecific features, such as temporal patterns of syscalls and correlation with network activities, will be integrated to provide a more comprehensive analysis of potential threats. Real-time detection capabilities will be enhanced to meet operational constraints, particularly in military applications.

## REFERENCES

- [1] William Steingartner and Darko Galinec. Cyber threats and cyber deception in hybrid warfare. Acta Polytechnica Hungarica , 18(3):25-45, 2021.
- [2] Paul Th´ eron and Alxander Kott. When autonomous intelligent goodware will fight autonomous intelligent malware: A possible future of cyber defense. In MILCOM 2019-2019 IEEE Military Communications Conference (MILCOM) , pages 1-7. IEEE, 2019.
- [3] Mariarosaria Taddeo, Tom McCutcheon, and Luciano Floridi. Trusting artificial intelligence in cybersecurity is a double-edged sword. Nature Machine Intelligence , 1(12):557-560, 2019.
- [4] Iqbal H Sarker, ASM Kayes, Shahriar Badsha, Hamed Alqahtani, Paul Watters, and Alex Ng. Cybersecurity data science: an overview from machine learning perspective. Journal of Big data , 7:1-29, 2020.
- [5] Farzad Nourmohammadzadeh Motlagh, Mehrdad Hajizadeh, Mehryar Majd, Pejman Najafi, Feng Cheng, and Christoph Meinel. Large language models in cybersecurity: State-of-the-art. arXiv preprint arXiv:2402.00891 , 2024.
- [6] Andrei Kucharavy, Zachary Schillaci, Lo¨ ıc Mar´ echal, Maxime W¨ ursch, Ljiljana Dolamic, Remi Sabonnadiere, Dimitri Percia David, Alain Mermoud, and Vincent Lenders. Fundamentals of generative large language models and perspectives in cyber-defense. arXiv preprint arXiv:2303.12132 , 2023.
- [7] Ramon Solo de Zaldivar, Alberto Huertas Celdr´ an, Jan von der Assen, Pedro Miguel S´ anchez S´ anchez, G´ erˆ ome Bovet, Gregorio Mart´ ınez P´ erez, and Burkhard Stiller. Malwspecsys: A dataset containing syscalls of an iot spectrum sensor affected by heterogeneous malware, 2022.
- [8] Neminath Hubballi, Santosh Biswas, and Sukumar Nandi. Sequencegram: n-gram modeling of system calls for program based anomaly detection. In 2011 Third International Conference on Communication Systems and Networks (COMSNETS 2011) , pages 1-10. IEEE, 2011.
- [9] Muhammad Ali, Stavros Shiaeles, Gueltoum Bendiab, and Bogdan Ghita. Malgra: Machine learning and n-gram malware feature extraction and detection system. Electronics , 9(11):1777, 2020.
- [10] Alberto Huertas Celdr´ an, Pedro Miguel S´ anchez S´ anchez, Chao Feng, G´ erˆ ome Bovet, Gregorio Mart´ ınez P´ erez, and Burkhard Stiller. Privacypreserving and syscall-based intrusion detection system for iot spectrum sensors affected by data falsification attacks. IEEE Internet of Things Journal , 2022.
- [11] Quentin Fournier, Daniel Aloise, and Leandro R Costa. Language models for novelty detection in system call traces. arXiv preprint arXiv:2309.02206 , 2023.
- [12] Gyuwan Kim, Hayoon Yi, Jangho Lee, Yunheung Paek, and Sungroh Yoon. Lstm-based system-call language modeling and robust ensemble method for designing host-based intrusion detection systems. arXiv preprint arXiv:1611.01726 , 2016.
- [13] Ansam Khraisat, Iqbal Gondal, Peter Vamplew, and Joarder Kamruzzaman. Survey of intrusion detection systems: techniques, datasets and challenges. Cybersecurity , 2(1):1-22, 2019.
- [14] Crispin Almodovar, Fariza Sabrina, Sarvnaz Karimi, and Salahuddin Azad. Can language models help in system security? investigating log anomaly detection using bert. In Proceedings of the The 20th Annual Workshop of the Australasian Language Technology Association , pages 139-147, 2022.
- [15] Song Chen and Hai Liao. Bert-log: Anomaly detection for system logs based on pre-trained language model. Applied Artificial Intelligence , 36(1):2145642, 2022.
- [16] Sreeraj Rajendran, Roberto Calvo-Palomino, Markus Fuchs, Bertold Van den Bergh, H´ ector Cordob´ es, Domenico Giustiniano, Sofie Pollin, and Vincent Lenders. Electrosense: Open and big spectrum data. IEEE Communications Magazine , 56(1):210-217, 2017.
- [17] Hammerzeit. BASHLITE. https://github.com/hammerzeit/BASHLITE, 2016. Last accessed: 15 April, 2024.
- [18] Nccgroup. The Tick - A simple embedded Linux backdoor. https: //github.com/nccgroup/thetick/, 2021. Last accessed: 15 April, 2024.
- [19] Error996. bedevil (bdvl). https://github.com/Error996/bdvl/, 2020. Last accessed: 15 April, 2024.
- [20] Jimmyly00. Ransomware PoC GitHub repository. https://github.com/ jimmy-ly00/Ransomware-PoC, 2020. Last accessed: 15 April, 2024.
- [21] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´ emi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771 , 2019.
- [22] Alberto Huertas Celdr´ an, Pedro Miguel S´ anchez S´ anchez, Miguel Azor´ ın Castillo, G´ erˆ ome Bovet, Gregorio Mart´ ınez P´ erez, and Burkhard Stiller. Intelligent and behavioral-based detection of malware in iot spectrum sensors. International Journal of Information Security , 22(3):541-561, 2023.