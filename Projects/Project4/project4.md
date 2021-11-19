# Project 4

Posted: November 19, 2021

Due: December 07, 2021


## Classification

### Gradient Descent

<p><strong>Problem 1</strong> <em>Implement the gradient descent algorithm (either batch or stochastic versions) for multiple linear regression. i.e., extend the version of the algorithm in the lecture notes to multiple parameters.</em></p>
<p>The gradient descent update equation for logistic regression is given by:</p>

<!--
<p><span class="math display">\[
\beta^{k+1} = \beta^k + \alpha \sum_{i=1}^{n} (y_i - p_i(\beta^k))\mathbf{x_i}
\]
-->
<img src="figs/eq1mod.png" alt="Equation 1" width="250"/>

<p>where (from the definition of log-odds):</p>

<!--
<p><span class="math display">\[
p_i(\beta^k) = \frac{e^{f_i(\beta^k)}}{1+e^{f_i(\beta^k)}}
\]</span></p>
-->
<img src="figs/eq2mod.png" alt="Equation 2" width="200"/>

<p>and
<!--
<span class="math inline">\(f_i(\beta^k) = \beta_0^k + \beta_1^k x_{i1} + \beta_2^k x_{i2} + \cdots + \beta_p^k x_{ip}\)</span>.</p>
-->

<img src="figs/eq3mod.png" alt="Equation 3" width="350"/></p>

<p><strong>Problem 2</strong> <em>Derive the above update equation</em>. Write the derivation in a markdown ipynb cell. </p>
<p><strong>Problem 3</strong> <em>Implement the gradient descent algorithm (either batch or stochastic versions) for multiple logistic regression.</em> i.e., modify your code in problem 1 for the logistic regression update equation.</p>
<p>Make sure you include in your submission writeup, which version of the algorithm you are solving (stochastic or batch), and make sure to comment your code to help us understand your implementation.</p>
<p><strong>Problem 4</strong> To test your programs, simulate data from the linear regression and logistic regression models and check that your implementations recover the simulation parameters properly.</p>
<p>Use the following functions to simulate data for your testing:</p>

<pre class="r"><code>
&#35;simulate data for linear regression

gen_data_x, gen_data_y = sklearn.datasets.make_regression(n_samples=100, n_features=20, noise = 1.5)

&#35;simulate data for logistic regression.  This is similar to linear, only now values are either 0 or 1.  
log_gen_data_x, dump_y = sklearn.datasets.make_regression(n_samples=100, n_features=20, noise = 1.5)
log_gen_data_y = [0 if i>0 else 1 for i in dump_y]}</code></pre>

<p>You can use this function as follows in your submission:</p>
<pre class="r"><code>
&#35;a really bad estimator
&#35;returns random vector as estimated parameters
dummy = np.ndarray([100, 20])
for index, row in enumerate(dummy):
    dummy[index] = np.random.normal(0, .1, 20)
plt.plot(gen_data_x, dummy)</code></pre>

<img src="figs/scatter.png" height=350>

<p>Include a similar plot in your writeup and comment on how your gradient descent implementation is working.</p>
</div>
<div id="try-it-out" class="section level2">

<h3>Try it out!</h3>

<ol style="list-style-type: lower-alpha">
<li><p>Find a small dataset on which to try out different classification (or regression) algorithms. </p></li>
<li><p>Choose <strong>two</strong> of the following algorithms:</p></li>
</ol>
<ol style="list-style-type: decimal">
<li>classification (or regression) trees,</li>
<li>random forests<br />
</li>
<li>linear SVM,</li>
<li>non-linear SVM</li>
<li>k-NN classification (or regression)</li>
</ol>
<p>and compare their prediction performance on your chosen dataset to your logistic regression gradient descent implementation using 10-fold cross-validation. Note: for those algorithms that have hyper-parameters, i.e., all of the above except for LDA, you need to specify in your writeup which model selection procedure you used.</p>
</div>



<div id="handing-in" class="section level2">

## Submission

Prepare a Jupyter notebook that includes answers for each Problem:


<ol style="list-style-type: decimal">
<li><p>For Problems 1 and 3 include your code in the writeup. Make sure they are commented and that the code is readable in your final writeup (e.g., check line widths).</p></li>
<li><p>For Problem 2, include the derivation of the gradient descent update in the writeup</p></li>
<li><p>For Problem 4, make sure you run the provided code and include the output in the writeup.</p></li>
<li><p>For the next section organize your writeup as follows:</p></li>
</ol>
<ol style="list-style-type: lower-alpha">
<li><p>Describe the dataset you are using, including: what is the outcome you are predicting and what are the predictors you will be using.</p></li>
<li><p>Include code to obtain and prepare your data as a dataframe to use with your three classification algorithms. In case your dataset includes non-numeric predictors, include the code you are using to transform these predictors into numeric predictors you can use with your logistic regression implementation.</p></li>
<li><p>Specify the two additional algorithms you have chosen in part (b), and for algorithms that have hyper-parameters specify the method you are using for model selection.</p></li>
<li><p>Include all code required to perform the 10-fold cross-validation procedure on your three algorithms.</p></li>
<li><p>Writeup the result of your 10-fold cross-validation procedure. Make sure to report the 10-fold CV estimate of each of the algorithms.</p></li>
</ol>

Submit to ELMS at Project 4 Assignment Submission and a .pdf to Gradescope.

## Group work

You are encouraged to work in small groups, but you must prepare your own writeup and submit it. Include the names of the peers who you worked with in the writeup.
