# MFE-ETP
<a name="readme-top"></a>

<p align="left">
    <a href='http://arxiv.org/abs/2210.03929'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://buzz-beater.github.io/assets/publications/2022_egotaskqa_nips/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://sites.google.com/view/egotaskqa'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

This repo contains code for our NeurIPS Datasets and Benchmarks 2024 paper:

[MFE-ETP: A Comprehensive Evaluation Benchmark for Multi-modal Foundation Models on Embodied Task Planning](https://)

/*[Baoxiong Jia](https://buzz-beater.github.io/), [Ting Lei](https://scholar.google.com/citations?user=Zk7Vxz0AAAAJ&hl=en), [Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/), [Siyuan Huang](https://siyuanhuang.com/)*/

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## File Structure and Usage
<details>
  <summary>File Structure and Usage</summary>
  <ol>
    <li>
      <a>data: benchmark data classified by capability aspect</a>
    </li>
    <li>
      <a>scrpts</a>
      <ul>
        <li><a>benchmark_predict_blip.py: inference script for BLIP-2 and InstructBLIP</a></li>
        <li><a>chat_gpt_api.py: encapsulating ChatGPT inference</a></li>
        <li><a>dataset_predict_gpt4v.py: inference script for ChatGPT(-4V)</a></li>
        <li><a>dataset_predict_llava.py: inference script for llava</a></li>
        <li><a>dataset_predict_minicpm.py: inference script for minicpm</a></li>
        <li><a>dataset_predict_minigpt4.py: inference script for minigpt4</a></li>
        <li><a>evaluate_gpt3.5_mp: using GPT-3.5 to evaluate prediction results with multithreading</a></li>
        <li><a>minigpt4_eval.yaml.py: configuration file for Minigpt4</a></li>
        <li><a>openai_cfg.json: configuration file for OpenAI api</a></li>
        <li><a>task_planning.py: Embodied Reasoning with GPT-4V</a></li>
      </ul>
    </li>
    <li><a>LICENSE: license file</a></li>
    <li><a>README.md</a></li>
  </ol>
</details>

## Getting Started

### Prerequisites

We recommend building a standalone Conda environment for the model you want to use. For example if you want to use llava-1.5:

  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)

2. Clone the repo

   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```

3. Install NPM packages

   ```sh
   npm install
   ```

4. Enter your API in `config.js`

   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## LICENSE
Our MFE-ETP benchmark is released under the MIT license. See the "LICENSE" file for additional details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Xian Fu - xianfu@tju.edu.cn

Min Zhang - min_zhang@tju.edu.cn

<p align="right">(<a href="#readme-top">back to top</a>)</p>


