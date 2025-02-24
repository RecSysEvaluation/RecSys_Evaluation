<!DOCTYPE html>
<html>
<head>

</head>
<body>


<h2>SessionRecGraphFusion framework</h2>

<h3>Introduction</h3>
<p align="justify">This reproducibility package was prepared for the paper titled "Who is the Winner in the Race of New Proposals?" and submitted to RecSys 2025. 
The results reported in this paper were achieved with the help of the SessionRecGraphFusion framework, which is built on the session-rec framework. Session-rec is a 
Python-based framework for building and evaluating recommender systems. It implements a suite of state-of-the-art algorithms and baselines for session-based and 
session-aware recommendation. More information about the session-rec framework can be <a href="https://rn5l.github.io/session-rec/index.html">found here.</a></p>
<h4>Compared algorithms</h4>
The following algorithms are compared in this study
<ul>
  <li>Session-based Recommendations with Recurrent Neural Networks (ICLR 2016)</li>
  <li> STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation (KDD 2018)</li>
  <li>Neural Attentive Session-based Recommendation (SIGIR'2018)</li>
  <li>Session-based recommendation with graph neural networks ( AAAI 2019) </li>
  <li>Global Context Enhanced Graph Neural Networks for Session-based Recommendation SIGIR 2020 </li>
  <li>Target Attentive Graph Neural Networks for Session-based Recommendation (SIGIR 2020) </li>
  <li>Graph Neighborhood Routing and Random Walk for Session-based Recommendation ICDM '21 </li>
  <li>COTREC: Self-Supervised Graph Co-Training for Session-based Recommendation (SIGIR 2021) </li>
  <li>Fusion of Latent Categorical Prediction and Sequential Prediction for Session-based Recommendation (Information Sciences-2021 IF: 8.10) </li>
</ul>

<h5>Required libraries to run the framework</h5>
<ul>
  <li>Anaconda 4.X (Python 3.5 or higher)</li>
  <li>numpy=1.23.5</li>
  <li>pandas=1.5.3</li>
  <li>torch=1.13.1</li>
  <li>scipy=1.10.1</li>
  <li>python-dateutil=2.8.1</li>
  <li>pytz=2021.1</li>
  <li>certifi=2020.12.5</li>
  <li>pyyaml=5.4.1</li>
  <li>networkx=2.5.1</li>
  <li>scikit-learn=0.24.2</li>
  <li>keras=2.11.0</li>
  <li>six=1.15.0</li>
  <li>theano=1.0.3</li>
  <li>psutil=5.8.0</li>
  <li>pympler=0.9</li>
  <li>Scikit-optimize</li>
  <li>tensorflow=2.11.0</li>
  <li>tables=3.8.0</li>
  <li>scikit-optimize=0.8.1</li>
  <li>python-telegram-bot=13.5</li>
  <li>tqdm=4.64.1</li>
  <li>dill=0.3.6</li>
  <li>numba</li>
</ul>
<h2>Installation guide</h2>  
<p>This is how the framework can be downloaded and configured to run the experiments</p>
  
<h5>Using Docker</h5>
<ul>
  <li>Download and install Docker from <a href="https://www.docker.com/">https://www.docker.com/</a></li>
  <li>Run the following command to "pull Docker Image" from Docker Hub: <code>docker pull shefai/session_rec_graph_fusion:latest</code>
  <li>Clone the GitHub repository by using the link: <code>https://github.com/Faisalse/SessionRecGraphFusion.git</code>
  <li>Move into the <b>SessionRecGraphFusion</b> directory</li>
  
  <li>Run the command to mount the current directory <i>SessionRecGraphFusion</i> to the docker container named as <i>session_rec_graph_container</i>: <code>docker run --name session_rec_graph_container  -it -v "$(pwd):/SessionRecGraphFusion" -it shefai/session_rec_graph_fusion:latest</code>. If you have the support of CUDA-capable GPUs then run the following command to attach GPUs with the container: <code>docker run --name session_rec_graph_container  -it --gpus all -v "$(pwd):/SessionRecGraphFusion" -it shefai/session_rec_graph_fusion:latest</code></li> 
<li>If you are already inside the runing container then run the command to navigate to the mounted directory <i>SessionRecGraphFusion</i>: <code>cd /SessionRecGraphFusion</code> otherwise starts the "session_rec_graph_container" and then run the command</li>
<li>Finally run this command to reproduce the results: <code>python run_config.py conf/in conf/out</code></li>
</ul>
<h5>Using Anaconda</h5>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <code>https://github.com/Faisalse/SessionRecGraphFusion.git</code></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into the <b>SessionRecGraphFusion</b> directory</li>
    <li>Run this command to create virtual environment: <code>conda create --name SessionRecGraphFusion python=3.8</code></li>
    <li>Run this command to activate the virtual environment: <code>conda activate SessionRecGraphFusion</code></li>
    <li>Run this command to install the required libraries for CPU: <code>pip install -r requirements_cpu.txt</code>. However, if you have support of CUDA-capable GPUs, 
        then run this command to install the required libraries to run the experiments on GPU: <code>pip install -r requirements_gpu.txt</code></li>
    
    
  </ul>
  <h4>Run algorithms to generate results reported in the main table</h4>
  
  <ul>
  <li>Finally run this command to reproduce the results: <code>python run_config.py conf/in conf/out</code></li> 
  </ul> 
  <h6>Run experiments with limited resources</h5>
  <p>To ensure the reproducibility of the results reported in the main table, we recorded the recommendation files for each considered dataset. You just need to run the following commands to reproduce the results.</p>
  <ul>
  <li>For the Diginetica dataset, run this command, and it will take approximately 160 seconds to reproduce the results. <code>python run_experiments_with_limited_resources.py --dataset diginetica</code>  </li>
  <li>For the Retailrocket dataset, run this command, and it will take approximately 70 seconds to reproduce the results. <code>python run_experiments_with_limited_resources.py --dataset retailrocket</code>  </li>
  <li>For the RSC15 dataset, run this command, and it will take approximately 30 seconds to reproduce the results. <code>python run_experiments_with_limited_resources.py --dataset rsc15</code>  </li>
  </ul>



  <h4>Run experiments to perform statistical analysis.</h4>
  <ul>
  <li>For the Diginetica dataset, run this command, and it will take approximately ten seconds to conduct statistical analysis. <code>python run_statistics_analysis.py --dataset diginetica</code>  </li>
  <li>For the Retailrocket dataset, run this command, and it will take approximately five seconds to conduct statistical analysis. <code>python run_statistics_analysis.py --dataset retailrocket</code>  </li>
  <li>For the RSC15 dataset, run this command, and it will take approximately two seconds to conduct statistical analysis. <code>python run_statistics_analysis.py --dataset rsc15</code>  </li>
  </ul>



  <h4>Run experiments for short, medium, and long sessions using the selected algorithms.</h4>
   <ul>
  <li>For the Diginetica dataset, run this command, and it will take approximately three seconds to reproduce the results. <code>python run_experiments_with_different_session_length.py --dataset diginetica</code>  </li>
  <li>For the Retailrocket dataset, run this command, and it will take approximately one seconds to reproduce the results. <code>python run_experiments_with_different_session_length.py --dataset retailrocket</code>  </li>
  <li>For the RSC15 dataset, run this command, and it will take approximately one second to reproduce the results. <code>python run_experiments_with_different_session_length.py --dataset rsc15</code>  </li>
  </ul>


</p>
</body>
</html>  

