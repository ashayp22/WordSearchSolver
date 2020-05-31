# Word Search Solver


requirements:

- node.js with dependencies
- python 3.6 with tensorflow 1.x and other libraries

Word Search Solver was my submission to [Congressional App Challenge 2019-2020](https://www.congressionalappchallenge.us/) for Illinois District 8. The project is a website that uses deep learning to solve word search puzzles. After the user has to take a picture of their puzzle, the image is sent to a server that uses image processing algorithms and a Convolutional Neural Network to read the image. A word-search solving algorithm is then applied to the converted puzzle and the results are sent back to the user, with the original image having highlighted words. The user also recieves definitions on every word they are looking for. This app was designed for students who struggle with reading and writing. The project tied for 2nd place at the District Level

## Getting Started

These instructions will get you a copy of the algorithm and running on your local machine for development and testing purposes.

### Prerequisites

Your machine needs to be compatible for running Java or C# Code. I recommend using the following IDEs or Engines that I have used the algorithm with:

Node.js
```
jMonkeyEngine
NetBeans
```
Python 3.6
```
Unity Game Engine
Visual Studio
```

### Installing

A step by step series of examples that tell you how to get a development env running

Download the zipped version of this repository and choose the C# version or Java version of the algorithm. Then, add the unzipped files to your project directory

Next, adjust the hyperparameters found in Settings.cs or Settings.java if you want to.
```
public static int NUM_AI = 40; //changes the number of agents to 40
```
You should now be ready to implement the algorithm in your project.



## Authors

* **Ashay Parikh** - (https://ashayp.com/)

## License

This project is licensed under the Gnu General Public License - see the [LICENSE.md](https://github.com/ashayp22/WordSearchSolver/blob/master/LICENSE) file for details


