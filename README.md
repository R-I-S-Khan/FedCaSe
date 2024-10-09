<h1> FedCaSe: A Caching and Scheduling Framework for Large-scale Federated Learning </h1>

**FedCaSe is a unified intelligent client
scheduling, data sampling, and caching solution for millions
of client devices in Federated Learning designed to address the I/O bottleneck resulting from heterogeneous limited memory**. 

FedCaSe efficiently caches client samples in-situ on limited on-device storage and
schedules client participation. FedCaSe boosts the I/O performance by exploiting a unique characteristic—the experience, i.e., relative impact on overall performance, of data samples and clients. FedCaSe utilizes this information in adaptive caching policies for sample placement inside the limited memory of edge clients. The framework also exploits the experience information to orchestrate the future selection of clients. FedCaSe's policies result in high local and global hit ratio for the clients, thus improving the training time for accuracy convergence.

**FedCaSe is built on FedScale, a scalable and extensible open-source federated learning (FL) engine and benchmark**.

FedScale ([fedscale.ai](https://fedscale.ai/)) provides high-level APIs to implement FL algorithms, deploy and evaluate them at scale across diverse hardware and software backends. 
FedScale also includes the largest FL benchmark that contains FL tasks ranging from image classification and object detection to language modeling and speech recognition. 
Moreover, it provides datasets to faithfully emulate FL training environments where FL will realistically be deployed.

## Getting Started

### Quick Installation (Linux)

You can simply run `install.sh`.

```
source install.sh # Add `--cuda` if you want CUDA 
pip install -r requirements.txt && pip install -e .
```

Update `install.sh` if you prefer different versions of conda/CUDA. Running **install.sh** should install all dependencies in a conda environment and also activate the `fedscale` conda environment on the bash terminal.

### Installation from Source (Linux/MacOS)

If you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) installed and cloned FedCaSe, here are the instructions.
```
cd fedcase

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
FEDCASE_HOME=$(pwd)
echo export FEDCASE_HOME=$(pwd) >> ~/.bashrc 
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
echo alias fedscale=\'bash $FEDCASE_HOME/fedscale.sh\' >> ~/.bashrc
conda init bash
. ~/.bashrc

conda env create -f environment.yml
conda activate fedscale
pip install -r requirements.txt && pip install -e .
```

Finally, install NVIDIA [CUDA 10.2](https://developer.nvidia.com/cuda-downloads) or above if you want to use FedScale with GPU support.


### Tutorials

Now that you have FedScale and FedCaSe installed, you can start exploring FedScale following one of these introductory tutorials.

1. [Explore FedScale datasets](./docs/Femnist_stats.md)
2. [Deploy your FL experiment](./docs/tutorial.md)
3. [Implement an FL algorithm](./examples/README.md)


## FedScale Datasets

***We are adding more datasets! Please contribute!***

FedScale consists of 20+ large-scale, heterogeneous FL datasets and 70+ various [models](./fedscale/utils/models/cv_models/README.md).

Each one is associated with its training, validation, and testing datasets. 
We acknowledge the contributors of these raw datasets. Please go to the `./benchmark/dataset` directory and follow the dataset [README](./benchmark/dataset/README.md) for more details.

The datasets can be downloaded using the following command:

```
./benchmark/dataset/download.sh download DATASET
```
## FedCaSe Runtime
FedCaSe Runtime is a scalable and extensible deployment built on FedSCale. 

### Hardware Dependencies

Running experiments do not mandate any special hardware. However, to run the experiments in a reasonable amount of time servers with fast Nvidia GPUs (e.g., A100/V100) or P100 GPUs are recommended. However, due to the scale of the experiments conducted in this study, it may not be feasible to reproduce it due to the large cost incurred. To give an estimate, even with advanced GPUs such as 6 P100 GPUs, it takes around 24 hours to run one benchmarking experiment and each figure in the paper consists of multiple such benchmarks. Nevertheless, to facilitate the reproducibility of the artifact, we will show how to run and reproduce the core results or contributions of this work.

### Software Dependencies

The FedCaSe framework's operation requires Python for core programming, Anaconda for package and environment management, and CUDA for GPU support in accelerated computing tasks. Essential packages and libraries required for FedCaSe are included in the **environment.yml** file and **requirements.txt** within the FedCaSe repository.

### Quick start (Experimentation Steps)

Before running FedCaSe, you need to first setup your storage and client nodes. Make sure the relevant ports are open and all of the nodes recognize each other by adding IP information in the `/etc/hosts` file. Set up public-private keys across the nodes and make sure you can ssh across the nodes without needing passwords. For example, if you have 3 nodes, you should make sure the following ssh accesses can be done first: node1->node2, node1->node3, node2->node1, node2->node3, node3->node1, node3-node2. 

Then you would need to install a filesystem, e.g. NFS server across all of the nodes you are going to use for simulating client training and mount them to a common directory `$HOME/client` Make sure you have read-write permissions enabled. You can follow this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-18-04) to set up an NFS mount on `$HOME/client`

Then you need to install redis.

```
cd $HOME
wget -P $HOME/ http://download.redis.io/releases/redis-6.0.1.tar.gz 
tar xzf redis-6.0.1.tar.gz
mv $HOME/redis-6.0.1 $HOME/redis-stable
cd $HOME/redis-stable
```

Change the `redis.conf` file. Some relevant parameters that you might want to change include the bind point, port, maxmemory, client-query-buffer-limit, io-threads, persistence, etc. Then start up redis servers in all of the client nodes.

```
screen -S redis-server $HOME/redis-stable/src/redis-server $HOME/redis-stable/redis.conf
```

You should then create a cluster. For example, using three nodes you would need to use the following command. The port would be `6379` unless you would like to change it.

```
$HOME/redis-stable/src/redis-cli --cluster create NODE1_IP:PORT NODE2_IP:PORT NODE3_IP:PORT --cluster-replicas 0
```

Now clone the fedcase repo, go to the main directory fedcase using the following command:

```
cd fedcase
```

First, edit **install.sh** script if necessary. Please, uncomment the parts relating to the installation of the Anaconda Package Manager, CUDA 10.2 if they are not already present on the servers. Note, if you prefer different versions of conda and CUDA, please check the comments in **install.sh** for details. 

To download FEMNIST dataset use the following command.
```
./benchmark/dataset/download.sh download femnist
```

Adjust the client device data files.
```
cp -r clientdata/cifar10 benchmark/dataset/data
cp -r clientdata/femnist benchmark/dataset/data
cp -r clientdata/device_info benchmark/dataset/data
```

Make necessary changes to the `benchmark/configs/femnist/conf.yml` file. If you want to test Oort, set `sample mode: oort`, `base_case: True`, `cachingpolicy: lru`, `subs: 0.0`, `clientsched: other`, `fedcaseimp: 0`, `fedcase: 0` for oort benchmarking. Change ps_ip and worker_ips to the host name of your nodes in the configuration file by cat \etc\hosts. For example, set 10.0.0.2:[4,4] as one of the worker_ips means launching 4 executors on each of the first two GPUs on 10.0.0.2 to train your model in a space/time sharing fashion.

### Running Baselines

Let's run a baseline first. 

Run a benchmarking experiment using the following code.
```
python docker/driver.py start $FEDCASE_HOME/benchmark/configs/femnist/conf.yml
```

Collect the necessary statistics from `$FEDCASE_HOME/femnist-logging` and `$HOME/client`

You can check the accuracy trend using the following command.

```
cat femnist-logging | grep "FL Testing"
```

Use the other necessary scripts provided in the GitHub repository to assistance in collecting statistics and plotting. For example, you can run the `calc_hits.py` to get the global RHR. 

```
python calc_hits.py
```

To find the cdf of local RHRs, you can use the following command.

```
python generateallclientio.py -d $HOME/client
```

To find the the round time required you can use the following command.
```
cat femnist-logging | grep "Training loss"
```

Then subtract the wall clock time of one round from the previous round to understand the round duration.

### Running FedCaSe

Now let's use fedcase techniques, you should set `fedcase: 1`, `base_case: False`, `sample_mode: random`, `cachingpolicy: fedcaseimp`, `clientsched: fedcase` in conf.yml for FedCaSe benchmarking. 

Use the following commands for fixing the parameters.

```
python docker/driver.py stop femnist
python fixtimestamp.py
find $HOME/client/ -type f ! -name "*_samples*" ! -name "*curr*" -delete
```

Run a fedcase benchmarking experiment using the following command.
```
python docker/driver.py start $FEDCASE_HOME/benchmark/configs/femnist/conf.yml
```

Use the same methods and scripts to understand RHR, accuracy, and round time duration.

## Repo Structure

```
Repo Root
|---- fedscale          # FedScale source code
  |---- core            # Core of FedScale service
  |---- utils           # Auxiliaries (e.g, model zoo and FL optimizer)
  |---- deploy          # Deployment backends (e.g., mobile)
  |---- dataloaders     # Data loaders of benchmarking dataset

|---- benchmark         # FedScale datasets and configs
  |---- dataset         # Benchmarking datasets

|---- examples          # Examples of implementing new FL designs
|---- docs              # FedScale tutorials and APIs
```

## References
Please read and/or cite as appropriate to use FedScale and FedCaSe code or data.

```bibtex
@inproceedings{fedcase2024,
  title={{FedCaSe}: A Caching and Scheduling Framework for Large-scale Federated Learning},
  author={XXX},
  booktitle={Submission'24},
  year={2024}
}
```
Please read and/or cite as appropriate to use FedScale code or data or learn more about FedScale.

```bibtex
@inproceedings{fedscale-icml22,
  title={{FedScale}: Benchmarking Model and System Performance of Federated Learning at Scale},
  author={Fan Lai and Yinwei Dai and Sanjay S. Singapuram and Jiachen Liu and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```