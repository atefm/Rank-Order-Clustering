// This code performs Rank Order Clustering on images based on histogram feautures

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <iostream>
#include <dirent.h>
#include <unordered_map>

using namespace cv;
using namespace std;

const int MAX_IMAGES_PER_FOLDER = 100;

// const int NUMBER_OF_TRACKLETS = 113;
const int NUMBER_OF_TRACKLETS = 35;
const int NUMBER_OF_FEATURES = 255*6;

const double MAX_DISTANCE = 0.2;
const double MERGE_THRESHOLD = 3;
const double CLUSTER_NORMALIZED_DISTANCE_THRESHOLD = 1.25;

const int TOTAL_IMAGES = 6000;

int actual_total_images = 0;

double feature_distances[TOTAL_IMAGES * TOTAL_IMAGES];
double all_features[NUMBER_OF_FEATURES * TOTAL_IMAGES];

double cluster_distances[NUMBER_OF_TRACKLETS * NUMBER_OF_TRACKLETS];

class Cluster
{
public:
	std::vector<int> TrackletNumbers;
	std::vector<int> ImageNumbers;
	Cluster(int tracklet_number, vector<int> imageNumbers){
		TrackletNumbers.push_back(tracklet_number);
		ImageNumbers = imageNumbers;
	};

	Cluster(){
	}
};

struct IndexDistace
{
	int Index;
	double Distance;

	IndexDistace(int index, double distance){
		Index = index;
		Distance = distance;
	}
};

struct by_distance
{
	bool operator()(IndexDistace const &a, IndexDistace const &b) { 
		return a.Distance < b.Distance;
	}
};

void ExtractFeatures(Mat image, double features[]){
	Mat split_channels[3];
	split(image, split_channels);

	int histSize = 255;
	float range[] = {1, 256};
	const float* histRange = { range };


	bool uniform = true;
	bool accumulate = true;

	int index = 0;

	for (int channelIndex = 0; channelIndex < 3; ++channelIndex){
		Mat top_half = split_channels[channelIndex](Range(0, split_channels[channelIndex].rows/2), Range::all());
		Mat top_hist;

		calcHist( &top_half, 1, 0, Mat(), top_hist, 1, &histSize, &histRange, uniform, accumulate);

		for (int i = 0; i < top_hist.rows; ++i){
			features[index] = top_hist.at<float>(i, 0);
			++index;
		}

		Mat bottom_half = split_channels[channelIndex](Range(split_channels[channelIndex].rows/2, split_channels[channelIndex].rows), Range::all());
		Mat bottom_hist;

		calcHist( &bottom_half, 1, 0, Mat(), bottom_hist, 1, &histSize, &histRange, uniform, accumulate);

		for (int i = 0; i < bottom_hist.rows; ++i){
			features[index] = bottom_hist.at<float>(i, 0);
			++index;
		}
	}
}

void copy_values_to_all_features(double features[]){
	int start_index = actual_total_images * NUMBER_OF_FEATURES;
	for (int i = 0; i < NUMBER_OF_FEATURES; i++){
		all_features[start_index + i] = features[i];
	}
}

void calculate_feature_distances(){
	int end = actual_total_images * NUMBER_OF_FEATURES;
	for (int i = 0; i < end; i += NUMBER_OF_FEATURES){
		int start_index = (i/NUMBER_OF_FEATURES) * actual_total_images;

		int count_i = i;
		double sum = 0.0;
		for (int j = i; j < end; ++j){
			sum += sqrt(all_features[count_i] * all_features[j]);

			++count_i;

			if (count_i >= i + NUMBER_OF_FEATURES){
				count_i = i;
				int fist_image_index = start_index/actual_total_images;
				int second_image_index = j/NUMBER_OF_FEATURES;

				feature_distances[start_index + (j/NUMBER_OF_FEATURES)] = 1 - sum;

				int inverse = (j/NUMBER_OF_FEATURES) * actual_total_images + (i/NUMBER_OF_FEATURES);

				feature_distances[inverse] = 1 - sum;

				sum = 0.0;
			}
		}
	}
}

double calculate_cluster_absolute_difference(Cluster cluster_a, Cluster cluster_b){

	// for (int feature_a =0; feature_a < cluster_a.ImageNumbers.size(); ++feature_a){
	// 	cout << cluster_a.ImageNumbers[feature_a] << endl;
	// }

	int index = cluster_a.ImageNumbers[0] * actual_total_images + cluster_b.ImageNumbers[0];

	double min_distance = feature_distances[index];

	for (int feature_a =0; feature_a < cluster_a.ImageNumbers.size(); ++feature_a){
		for (int feature_b =0; feature_b < cluster_b.ImageNumbers.size(); ++feature_b){
			double distance = feature_distances[cluster_a.ImageNumbers[feature_a] * actual_total_images + cluster_b.ImageNumbers[feature_b]];
			if (distance < min_distance){
				min_distance = distance;
			}
		}
	}

	return min_distance;
}

void calculate_cluster_distances(std::vector<Cluster> clusters){
	for (int i = 0; i < clusters.size(); ++i){
		for (int j = 0; j < clusters.size(); ++j){
			cluster_distances[i * NUMBER_OF_TRACKLETS + j] = calculate_cluster_absolute_difference(clusters[i], clusters[j]);
		}
	}
}

bool hasEnding (std::string const &fullString, std::string const &ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

vector<string> getListOfFilesWithEnding(string directory, string ending){
	DIR * dir;

	struct dirent *ent;

	std::vector<string> imageNamesForTracklet;

	if ((dir = opendir(directory.c_str())) != NULL){

		while ((ent = readdir(dir)) != NULL){
			if (!hasEnding(ent->d_name, ending)){
				continue;
			}

			imageNamesForTracklet.push_back(ent->d_name);
		}

		closedir(dir);
	}

	return imageNamesForTracklet;

}

void divide_all_by_max(double features[]){
	double max_val = 0.;

	for (int i =0; i < NUMBER_OF_FEATURES; ++i){
		max_val = std::max(max_val, features[i]);
	}

	for (int i =0; i < NUMBER_OF_FEATURES; ++i){
		features[i] = features[i] / max_val;
	}
}

void normalize(double features[]){
	double sum_of_features = 0;

	for (int i =0; i < NUMBER_OF_FEATURES; ++i){
		sum_of_features += features[i];
	}

	for (int i =0; i < NUMBER_OF_FEATURES; ++i){
		features[i] = features[i] / sum_of_features;
	}
}

double calculate_feature_absolute_difference(double feature_set1[], double feature_set2[]){
	//Calculates the Byattacharyya distance of two histograms.
	double sum = 0;
	for (int i =0; i < NUMBER_OF_FEATURES; ++i){
		sum += sqrt(feature_set1[i] * feature_set2[i]);
	}
	return 1 - sum;
}


void calculate_cluster_ranks(int clusters, std::unordered_map<int, std::unordered_map<int, int>> &cluster_ranks, std::unordered_map<int, std::unordered_map<int, int>> &rank_to_cluster){
	for (int i = 0; i < clusters; ++i)
	{
		std::vector<IndexDistace> neighbors;

		for (int j=0; j < clusters; ++j){
			neighbors.push_back(IndexDistace(j, cluster_distances[i * NUMBER_OF_TRACKLETS + j]));
		}

		std::sort(neighbors.begin(), neighbors.end(), by_distance());

		for (int rank=0; rank < clusters; ++rank){
			IndexDistace item = neighbors[rank];

			cluster_ranks[i][item.Index] = rank;
			rank_to_cluster[i][rank] = item.Index;
		}
	}
}

double calcuate_asymmetric_rank_order_distance(std::unordered_map<int, std::unordered_map<int, int>> cluster_ranks, std::unordered_map<int, std::unordered_map<int, int>> rank_to_cluster, int cluster_a, int cluster_b){
	int iterate_till = cluster_ranks[cluster_a][cluster_b];

	if (iterate_till > 20){
		return 1000000000;
	}

	double sum = 0.0;

	for (int i=0; i < iterate_till; i++){
		sum += cluster_ranks[cluster_b][rank_to_cluster[cluster_a][i]];
	}

	return sum;
}

double calculate_cluster_rank_order_distance(std::unordered_map<int, std::unordered_map<int, int>> cluster_ranks, std::unordered_map<int, std::unordered_map<int, int>> rank_to_cluster, int cluster_a, int cluster_b){
	int denom = min(cluster_ranks[cluster_a][cluster_b], cluster_ranks[cluster_b][cluster_a]);

	// cout << "denom: " << denom << endl;

	int dist1 = calcuate_asymmetric_rank_order_distance(cluster_ranks, rank_to_cluster, cluster_a, cluster_b);
	int dist2 = calcuate_asymmetric_rank_order_distance(cluster_ranks, rank_to_cluster, cluster_b, cluster_a);

	return (dist1 + dist2) * 1./denom;
}

double calculate_cluster_level_normalized_distance(std::vector<Cluster> clusters, int cluster_a, int cluster_b, std::unordered_map<int, std::unordered_map<int, int>> rank_to_cluster){
	int num_neighbors = min(20, actual_total_images);

	double overall_sum = 0;

	for (int i=0; i<clusters[cluster_a].ImageNumbers.size(); ++i){

		double feature_sum = 0;

		int image_number = clusters[cluster_a].ImageNumbers[i];

		for (int k=0; k<num_neighbors; ++k) {
			feature_sum += feature_distances[image_number * actual_total_images + rank_to_cluster[image_number][k]];
		}

		double feature_average = feature_sum / num_neighbors;

		overall_sum += feature_average;
	}

	for (int i=0; i<clusters[cluster_b].ImageNumbers.size(); ++i){

		double feature_sum = 0;

		int image_number = clusters[cluster_b].ImageNumbers[i];
		
		for (int k=0; k<num_neighbors; ++k) {
			feature_sum += feature_distances[image_number * actual_total_images + rank_to_cluster[image_number][k]];
		}

		double feature_average = feature_sum / num_neighbors;

		overall_sum += feature_average;
	}

	int total_images_in_clusters = clusters[cluster_a].ImageNumbers.size() + clusters[cluster_b].ImageNumbers.size();

	double phi = (1.0/total_images_in_clusters) * overall_sum;

	double cluster_dist = cluster_distances[cluster_a * NUMBER_OF_TRACKLETS + cluster_b];

	double normalized_distance = (1/phi) * cluster_dist;

	return normalized_distance;
}



int main(){

	std::vector<Cluster> clusters;

	for (int tracklet_folder_index = 0; tracklet_folder_index <= NUMBER_OF_TRACKLETS; ++tracklet_folder_index) {
		string folder_name = "./tracklets_perfect/" + to_string(tracklet_folder_index) + "/";
		// string folder_name = "./tracklets_ch4_50_min_new/" + to_string(tracklet_folder_index) + "/";

		std::vector<string> imagesInFolder = getListOfFilesWithEnding(folder_name, ".jpg");

		if (imagesInFolder.size() < 20) {
			// cout << "skipping: " << tracklet_folder_index << endl;
			continue;
		}

		if (imagesInFolder.size() > MAX_IMAGES_PER_FOLDER) {
			std::vector<string> selectedImages;

			float increment = imagesInFolder.size() / MAX_IMAGES_PER_FOLDER;

			float current_index_float = 0.;

			while (selectedImages.size() < MAX_IMAGES_PER_FOLDER) {
				int current_index = (int) round(current_index_float);

				selectedImages.push_back(imagesInFolder[current_index]);

				current_index_float += increment;
			}

			imagesInFolder = selectedImages;
		}

		std::vector<int> cluster_feature_indicies;

		// for(auto imageName : imagesInFolder) { //this causes a seg fault for -02 optimization for some reason
		for (int i = 0; i < imagesInFolder.size(); ++i) {

			// if (actual_total_images >= 9){
			// 	break;
			// }
			string imageName = imagesInFolder[i];
		    // cout << imageName << "\n "; // this will print all the contents of *features*

			string imagePath = folder_name + imageName;

			// Mat image = imread(imagePath);
			// imshow("Display", image);
			// waitKey(0);
			Mat image = imread(folder_name + imageName);

			double* features = new double[NUMBER_OF_FEATURES] ;

			ExtractFeatures(image, features);

			divide_all_by_max(features);

			normalize(features);

			copy_values_to_all_features(features);

			cluster_feature_indicies.push_back(actual_total_images);

			++actual_total_images;
		}

		clusters.push_back(Cluster(tracklet_folder_index, cluster_feature_indicies));

		// for (int i=0; i<cluster_feature_indicies.size(); ++i){
		// 	cout << cluster_feature_indicies[i] << " ";
		// }
		// return 0;

		// cout << tracklet_folder_index << "\n";
	}

	// cout << "num features: " << actual_total_images << "\n";

	// std::unordered_map<int, std::vector<IndexDistace>> neighbors;
    // std::unordered_map<int, std::unordered_map<int, int>> feature_rank_to_index;

	calculate_feature_distances();

	bool merged = true;

	while (merged) {
		merged = false;

		calculate_cluster_distances(clusters);

		std::unordered_map<int, std::unordered_map<int, int>> cluster_ranks;
		std::unordered_map<int, std::unordered_map<int, int>> rank_to_cluster;

		calculate_cluster_ranks(clusters.size(), cluster_ranks, rank_to_cluster);


		std::unordered_map<int, int> new_cluster_numbers;

		for (int i = 0; i < clusters.size(); ++i) {
			new_cluster_numbers[i] = -1;
		}

		int new_cluster_index = 0;

		for (int i=0; i < clusters.size(); ++i){
			for (int j=i+1; j < clusters.size(); ++j){
				double cluster_rank_order_distance = calculate_cluster_rank_order_distance(cluster_ranks, rank_to_cluster, i, j);
				// cout << "cluster rank order  distance " << i << ", " << j << ": " << cluster_rank_order_distance << endl;

				double cluster_normalized_distance = calculate_cluster_level_normalized_distance(clusters, i, j, rank_to_cluster);
				// cout << "cluster normalized distance " << i << ", " << j << ": " << cluster_normalized_distance << endl;

				if (cluster_rank_order_distance < MERGE_THRESHOLD and cluster_normalized_distance < CLUSTER_NORMALIZED_DISTANCE_THRESHOLD){
					merged = true;

					if (new_cluster_numbers[i] == -1 && new_cluster_numbers[j] == -1){
						new_cluster_numbers[i] = new_cluster_index;
						new_cluster_numbers[j] = new_cluster_index;
						++new_cluster_index;
					} else if (new_cluster_numbers[i] == -1) {
						new_cluster_numbers[i] = new_cluster_numbers[j];
					} else if (new_cluster_numbers[j] == -1) {
						new_cluster_numbers[j] = new_cluster_numbers[i];
					}
				} else {
					// cout << "did not merge " << i << " and " << j << endl;
				}
			}
		}

		std::vector<Cluster> new_clusters;

		for (int c_number = 0; c_number < new_cluster_index; ++c_number){
			Cluster merged_cluster = Cluster();

			for (int old_cluster=0; old_cluster < clusters.size(); ++old_cluster){
				if (new_cluster_numbers[old_cluster] == c_number){
					for (int i=0; i < clusters[old_cluster].TrackletNumbers.size(); ++i){
						merged_cluster.TrackletNumbers.push_back(clusters[old_cluster].TrackletNumbers[i]);
					}

					for (int i=0; i < clusters[old_cluster].ImageNumbers.size(); ++i){
						merged_cluster.ImageNumbers.push_back(clusters[old_cluster].ImageNumbers[i]);
					}
				}
			}

			new_clusters.push_back(merged_cluster);
		}

		for (int old_cluster=0; old_cluster < clusters.size(); ++old_cluster){
			if (new_cluster_numbers[old_cluster] == -1){
				new_clusters.push_back(clusters[old_cluster]);
			}
		}

		clusters = new_clusters;

		cout << "Clusters: \n";

		for (int i =0; i < clusters.size(); ++i){
			for (int j=0; j< clusters[i].TrackletNumbers.size(); ++j){
				cout << clusters[i].TrackletNumbers[j] << " ";
			}
			cout << "\n";
		}
		cout << "##########################################################\n";
	}

	return 0;
}