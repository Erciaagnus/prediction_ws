#include <ros/ros.h>
#include <onedayalg/onedayalg.h>
#include <msgs/dataset_array_msg.h>
#include <msgs/map_array_msg.h>

int main (int argc, char **argv)
{
	ros::init(argc, argv, "supervisor");
	ros::NodeHandle node("~");

	ros::AsyncSpinner spinner(1);
  	spinner.start();

	Supervisor sup;
 	 
	sup.pub1 = node.advertise<msgs::dataset_array_msg>("/result",1);
	

	ros::Subscriber sub1 = node.subscribe("/map_data", 1,             &Supervisor::callback_map, &sup);
	ros::Subscriber sub2 = node.subscribe("/history", 1,             &Supervisor::callback_dataset, &sup);
	 
	ros::waitForShutdown();
}
