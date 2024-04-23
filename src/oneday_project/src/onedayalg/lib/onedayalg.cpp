#include <onedayalg/onedayalg.h>
#include <stdio.h>
#include <stdint.h>
#include <signal.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <msgs/dataset_array_msg.h>
#include <msgs/map_array_msg.h>



Supervisor::Supervisor()
{	
	
	
}


void Supervisor::callback_map(const msgs::map_array_msg& data){
	
	map_msg = data;
	
}


void Supervisor::callback_dataset(const msgs::dataset_array_msg& data){
	
	dataset_msg = data;
	
	msgs::dataset_array_msg results;
	results.time = ros::Time::now();

	for(int i=0 ; i<dataset_msg.data.size() ; i++){

		msgs::dataset_msg result;

		// To Do 
        // 각 vehicle마다 self.vehicles 안에 위치한 정보를 통해서 과거 10개 step(0.5s)의 관측 값을 바탕으로,
        // 현재 차량의 차선 변경 의도를 예측하기.
        // 그리고, result_temp 변수에 LC 이면 1, LK이면 0 값을 대입하여 결과를 확인해 봅시다.

        
		int result_temp = 1;

		result.id = i;
		result.mode = result_temp;

		results.data.push_back(result);


	}

	pub1.publish(results);

}






