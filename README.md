# Machine_learning_Naive_Sharding
In this implementation, I prove that the Naive Sharding algorithm executes in a lower number of iterations as opposed to the random. I find the accuracy score for each and the time taken for the algorithms to execute. 

The Naïve Sharding Centroid Initialization Method is another way of initializing centers as opposed to the random initialization of the centers in the K means algorithm in which the k points are usually randomly chosen as cluster centroids. Because the initialization of clusters is random, this can lead to an increased number of required clustering iterations to reach convergence. This is also accompanied by a greater overall runtime especially for huge data sets. On the other hand, the naïve sharding algorithm relies on the calculation of a composite summation value which is then used to sort the instances of the data set. The sharding algorithm is expected to execute quicker than the random centroid initialization as it does not depend on the data complexity and executes in linear time. 

In my implementation, I prove that the sharding algorithm executes in a lower number of iterations as opposed to the random. I also print the accuracy score for each, and the time taken for the algorithms to execute.  

It is clear that through the implementation of this algorithm, the naïve sharding algorithm only took 4 iterations to converge (less than the random initialization). It also took a shorter amount of time to converge as reflected by the time it took to converge. 

This is due because the naïve sharding algorithm executes linearly and does not depend on the complexity of the data set as opposed to the random initialization. The time needed for randomly initializing the centroids can grow super linearly. A random initialization of points can take a long time to converge since the distance between the center and the data points has to be calculated over and over again until convergence.
