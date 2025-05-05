# ai6127-assignment-3-solved
**TO GET THIS SOLUTION VISIT:** [AI6127 Assignment 3 Solved](https://www.ankitcodinghub.com/product/ai6127-assignment-3-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;95313&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;AI6127 Assignment 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
1 Question One

The Transformer [12] has been the dominant encoder/decoder architecture for many NLP tasks in recent years. With the access to larger GPU/TPU memory and its distributed computational capability, researchers have trained transformers with millions [9, 4, 8, 3], billions [2, 1, 10, 7] and trillions [5] of parameters. Some work [13, 6, 14] has been done on scaling the self/cross attention further. However, the basic structure of the model widely remains same. A taxonomy of the different modifications of the transformer architecture is shown in Figure 2 for those interested.

In this assignment, we will dive deep into the original transformer architecture proposed in [12]. In general, the transformer utilizes self/cross-attention with multiple heads and combines them via a regular feed-forward network. In the original base transformer, the basic hyperparameters are,

• Number of layers, N = 6

• Feature dimension for each token, dmodel = 512

1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Figure 1: Transformer Architecture [12] • Number of heads, h = 8

• Key and value representation size, dk = dmodel = 64 h

• Hidden representation size of the feed-forward layer, ffn_dim = 1024

Answer the following questions:

(a) Assume you have only one layer (N = 1) in the transformer.

<ul>
<li>Calculate the total number of parameters in the transformer encoder layer based on the
remaining default hyper-parameters mentioned above.
</li>
<li>Now increase the number of layers one by one up to 12 and show how the number of
parameters increases compared to the number of layers N.
</li>
<li>For each of the previous calculations, increase the token representation size of dmodel from 512 to 1024 and 2048, and show how the number of parameters increases compared to the number of layers.</li>
<li>Discuss how the total number of parameters changes with respect to the hyper-parameter values based on your experiments.</li>
<li>Implementation guide : You can use torch.nn.TransformerEncoderLayer to implement the transformer layer and calculate the number of parameters.
(b) Do you agree/disagree with the following statement (provide necessary arguments to support your stand):

“Multi-head attention works as an ensemble of heads in the transformer architecture.” 2
</li>
</ul>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
Figure 2: Taxonomy of Eﬀicient Transformer Architectures. [11]

(c) How is the decoder training and decoder inference different in the transformer architecture,

compared to recurrent models like LSTM/GRU?

<ol start="4">
<li>(d) &nbsp;How does the transformer architecture capture sequence information despite having no sequen- tial probability calculation like RNN?</li>
<li>(e) &nbsp;Follow the example code here. You will find a TransformerModel class in the model.py file which implements the TransformerEncoder class from the torch.nn library. You can also use the Huggingface library to train a language model. For training a language model with Huggingface, follow the tutorial here. We recommend writing the transformer encoder or decoder from scratch following the tutorial from Alexander Rush and integrating it into the project. Apart from that you may also look into popular library implementations (fairseq, OpenNMT-py, huggingface) and borrow different modules from there to implement your Transformer architecture.</li>
</ol>
• Train the word language model with wikitext (data is given in the repository). Train the model for 10 epochs. Select the best model based on development set perplexity. Report the perplexity of the test set.

</div>
</div>
<div class="layoutArea">
<div class="column">
• You may perform hyper-parameter search on the model based on the hyper-parameters (N, dmodel, h, dk, ffn_dim) discussed above in the assignment.

</div>
</div>
<div class="layoutArea">
<div class="column">
• What happens when you do not perform scaling (no ?dk in the softmax in equation 1 in [12]) in the attention head? Report the effect of scaling on perplexity.

</div>
</div>
<div class="layoutArea">
<div class="column">
Note: If you are using a third party library for implementing the Transformer model (i.e. Huggingface), be careful about the model parameters. You may have to handle the parameters differently in different libraries.

3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
References

<ol>
<li>[1] &nbsp;Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhari- wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020.</li>
<li>[2] &nbsp;Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhari- wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agar- wal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCan- dlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.</li>
<li>[3] &nbsp;Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wen- zek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440–8451, Online, July 2020. Association for Computational Linguistics.</li>
<li>[4] &nbsp;Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.</li>
<li>[5] &nbsp;William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and eﬀicient sparsity, 2021.</li>
<li>[6] &nbsp;Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention, 2020.</li>
<li>[7] &nbsp;Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021.</li>
<li>[8] &nbsp;Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach, 2019.</li>
<li>[9] &nbsp;Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.</li>
<li>[10] &nbsp;Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.</li>
<li>[11] &nbsp;Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Eﬀicient transformers: A survey, 2020.</li>
<li>[12] &nbsp;Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. CoRR, abs/1706.03762, 2017.</li>
<li>[13] &nbsp;Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity, 2020.
4
</li>
</ol>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
[14] Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences, 2021.

</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
