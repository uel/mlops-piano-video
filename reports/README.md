---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 38 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s240554, s240493---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used the third-party framework Denoising Diffusion Probabilistic Model from the package denoising-diffusion-pytorch ([link](https://github.com/lucidrains/denoising-diffusion-pytorch)) which is an implementation of a model by the same name in Pytorch. We used the package to create the model architecture and to run the training and the predict steps of the model. The package uses a Trainer class to help build and train the model. The sample method is used to generate an image from saved the saved model after training and the method name is sample().---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- We created the project environment using conda. We initiated an empty repository and added the necessary packages as when was needed. We noted down the package versions that we were installing and added them to the requirements.txt file. An exact copy of our environment can be obtained by running `pip install -r requirements.txt` in the root of the repo folder in a newly created environment. ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- Yes, we used the cookiecutter template provided along with the course. 

The train and predict script can be found in piano_video folder. The config.yaml file contains the config files used by Hydra to run the training script and predict script. We haven't used the piano_video/data and piano_video/models folder since we were generating out our data to be input to the Denoising Diffusion Probabilistic Model and the model was part of the denoising-diffusion-pytorch package. The output of wandb will be generated in a wandb folder in the root. The reports from each run is output into a timestamped folder in the reports folder of the root. The output from tenorboard will be saved in those folders. We have added a function_deployment folder to save the Cloud Function service and an app folder to save the FastAPI app which could be used for the deployment of the model (to be done). The other folders we haven't used are docs and notebooks. ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- We have not implemented any code quality rules or format since most of the code in the training and predict step came from the denoising-diffusion-pytorch package. We have added comments in the train and predict script to convey the intention of the code. We did not use docstrings since the code was mostly part of the denoising-diffusion-pytorch package. We didn't use ruff to format the code since again the code was mostly part of the package. ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- In total we have implemented 4 tests. First we test the instantiation of the model (U-Net backbone and diffusion model) and the Trainer class. Then we also test one iteration of the training loop and check if the weigths are updated succesfully. Finally we try to load the model from a file, generate an image and check if the image has correct size. ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- Code coverage of 100% doesying to fill some questions which are empty.n't guarantee a perfect model or perfect training or perfect data. A project with 100% code coverage can result in a faulty prediction and hence not error free.  ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- Yes, each of us used our our branches to work on the projects individually. The basic workflow was that each of us made our own branch to work on our parts of the project. The branches were mostly used to implement new features or changes. These changes once found to be working for others were pulled and merged to the main branch. We used pull request to merge branches into main and update the main branch. We used merge and rebase to keep our branches updated along with main. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- Initially we used a local copy of the data to work on the training and predict script. The data was then uploaded to a  google cloud bucket. The data was then versioned using dvc. When needed the data was pulled from the google cloud bucket using `dvc pull`. Since we did not work with different data or changing data, the data versioning was not than important. But this project will rely on dvc in the future once we start using larger or different datasets. Currently, we use images of piano video but that can be changed to something else and dvc could be used to tag and version control data then. ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- We have organized our CI into 2 separate files. The first file is used for unittesting. The testing also requires data so we first install dvc to be able to download them. The data are saved in a Google Cloud Bucket. We first authenticate using a JSON key which is saved as a GitHub secret and then we pull the data. Then the unit tests are run for the latest version of Ubuntu and Python 3.11. Besides running the unit tests we also calculate the code coverage and automaticaly update code coverage number displayed in our README. Thanks to this our README displays the current code coverage after each push. [Link to the workflow](https://github.com/uel/mlops-piano-video/blob/konarfil/.github/workflows/tests.yml)
Our second workflow file handles automatic deployment and ensures the most recent version of our code is deployed each time we push it to our repository. As in the previous case it authenticates using JSON key to access Google Cloud. It then uses [GitHub Action](https://github.com/google-github-actions/deploy-cloud-functions.git) to deploy a Cloud function. The Python script for the function is loaded from the function_deployment folder inside our repository. The function accepts a HTTP request, generates a random image of a piano and returns a HTML page with the image which can be displayed in user's browser. ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- The parameters for the denoising-diffusion-pytorch model was taken out of the training and predict script and moved into the config.yaml file located in piano_video/config folder. The config file contains all the necessary parameters and hyperparameters that are necessary to run the model including the torch seed. The config file was input using hydra package. To run an experiment adjust the config file as needed (change the necessary parameter eg: batch size) then run train_model.py using `python train_model.py`---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- We used Hydra config files to set all parameters. The config files are saved in Git commits so it is possible to return to previously used parameters. We also used Weights and Biases for logging so that information about experiments is saved. ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- The only metric that we have tracked for this project is loss during the training. The loss was only logged, to show the usage with  wandb and tensorboard, and not used to improve the model. The wandb [sreenshot](figures/wandb-screenshot.png) and tensorboard [screenshot](figures/tb-screenshot.png) are attached to show the loss logging.---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

---Each group member used debugging tools they were most familiar with but generally we used simple print statements as well as standard debugging tools offered by out IDEs (breakpoints, stepping through code, ...). To understand and solve errors which we encountered we also leveraged modern tools such as ChatGPT. We did not run profiling of our code the most computationally heavy parts of our project were handled by external library and because of that we did not see much space for performance improvement.---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- We used 
1. Cloud Storage - to store the data input for training and to save the model after training. The model was then taken from a bucked and served to a Cloud Function for deployment.
2. Cloud Build - used to build docker images from the github repository in a continuos manner using the trigger for the autobuild as the push to main branch of our repo and the build is done using a cloudbuild.yaml file. Forms part of CI.
3. Artifact/Container Registry - the docker images were saved in the Artifact Registry after the images were built using Cloud Build. The images could be accessed from the registry to initiate a new VM to do computations.
4. Compute Engine - was used to obtain a VM to perform training on the model.
7. Vertex AI - a better alternative for training the model in the cloud. Could start custom VMs, train the model, finish the training and close the created VM. We used Vertex AI to train the model.
5. Cloud Functions - we used a simple Cloud Function to deploy the model. The cloud function accessed the saved model from the bucket and displayed the image that was generated from the trained model.
6. Cloud Run - we made an app using fastapi to enable a user to access the model. The objective of the using cloud run is to auto build the predict docker image and then deploy it automatically as well. The cloud run will use the same trigger as the cloud build i.e. pushing to the main branch of the repo. ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- Image of GCP bucket - [1](figures/bucket-1.png), [2](figures/bucket-2.png), [3](figures/bucket-3.png) ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- Image of docker images for training generated as part of Cloud Build trigger from pushing to main branch - [image](figures/cr-2.png) ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- Image of Cloud build - [image](figures/cloud-build.png) ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- We deployed our model using Cloud Functions. For this we use a Python script which loads our model from a Google Cloud bucket, runs inference to generate an image and returns a HTML page containing the image which can be easily displayed by user's browser. We also created GitHub workflow which ensures that the Cloud function is updated every time we push new version of the function to the repository.  ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- We did not implement monitoring as our final model is quite simple since we spent most of the time on the MLOps part of the project. One metric we could potentially measure is Fréchet inception distance which compares images generated by the model with the real images and can be used to assess quality of the generated ones. Another way to monitor the model could be to deploy it and let users rate quality of the generated images. This would, however, require more sofisticated infrastructure. ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- Image of architecture - [image](figures/diagram.png) As the diagram shows we made use of multiple tools which the course made us familiar with. The starting point of the diafram is the local repository saved on our computers. To backup and share the code we used GitHub repository. In the repository we also setup GitHub actions workflow. The first workflow takes care of automatic unit testing. Later we added a second workflow which automatically deploys a Cloud function. The training of the model was done in two ways. First we tested the training on our local GPU. Later on we also trained the model in Cloud on more data. We used config files to save parameters of our experiments. The parameters were loaded using Hydra. We also used Weights and Biases for logging during training. Both data and the trained model are saved in Cloud storage. For data loading and version control we used DVC. The trained model is finally deployed as a Cloud function. As mentioned the code of the function is automatically deployed using GitHub Actions. The function itself loads the trained model from the Cloud storage. It accepts a HTTP request from the user and returns an image generated by the model in form of HTTP page. ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---