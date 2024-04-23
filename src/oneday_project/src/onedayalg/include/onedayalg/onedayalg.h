#ifndef ONEDAYALG_H
#define ONEDAYALG_H

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <cerrno>
#include <cstring>

#include <string.h> 
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <ros/ros.h>
#include <math.h>
#include <tuple>
#include <msgs/dataset_array_msg.h>
#include <msgs/map_array_msg.h>


#include <math.h>
#include <numeric>
#include <ctime>
#include <omp.h>
 
 

class Supervisor{

	public:
		ros::Publisher pub1;

				

		msgs::map_array_msg map_msg = msgs::map_array_msg();
		msgs::dataset_array_msg dataset_msg = msgs::dataset_array_msg();
		
		Supervisor();

		void callback_map(const msgs::map_array_msg &data);
		void callback_dataset(const msgs::dataset_array_msg& data);
		

};

#endif
