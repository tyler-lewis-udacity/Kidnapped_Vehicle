/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// create a random engine generator
default_random_engine gen;

// min allowable value
double epsilon = 0.000001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of particles
	num_particles = 100;

	// standard deviations
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// create normal distributions
	normal_distribution<double> N_x(x, std_x);// standard deviation x
	normal_distribution<double> N_y(y, std_y);// standard deviation y
	normal_distribution<double> N_theta(theta, std_theta);// standard deviation theta

	// initialize particles
	for(int i=0; i < num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1);
	}
	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// create normal distributions (0 mean)
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {

    // calculate new state
    if (fabs(yaw_rate) < epsilon) { //if the yaw_rate is nearly zero... 
      particles[i].x += velocity * delta_t * cos(particles[i].theta);// new x-state
      particles[i].y += velocity * delta_t * sin(particles[i].theta);// new y-state
      // theta stays the same
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
  // add noise
  particles[i].x += N_x(gen);
  particles[i].y += N_y(gen);
  particles[i].theta += N_theta(gen);
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	unsigned int n_observations = observations.size();
  unsigned int n_predictions = predicted.size();

  for(unsigned int i=0; i < n_observations; i++) { 

    // initialize minimum distance
    double min_distance = numeric_limits<double>::max();

    // initialize map id
    int map_id = -1;

    for(unsigned int j=0; j < n_predictions; j++) {

      double x_distance = observations[i].x - predicted[j].x;
      double y_distance = observations[i].y - predicted[j].y;
      double distance = x_distance*x_distance + y_distance*y_distance;

      // ensure that distance is not too small
      if(distance < min_distance) {
        min_distance = distance;
        map_id = predicted[j].id;
      }
    }

    // Update the observation identifier.
    observations[i].id = map_id;
  }	
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
  for(int i = 0; i < num_particles; i++) {

    // get the particle x, y coordinates
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    // collect landmarks that are in range of the sensor
    vector<LandmarkObs> in_range;
    for(unsigned int j=0; j < map_landmarks.landmark_list.size(); j++) {

      // get landmark coordinates
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      // add in-range landmarks to in_range
      if(fabs(landmark_x - x) <= sensor_range && fabs(landmark_y - y) <= sensor_range) {
        in_range.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    // transform vehicle coordinates to global map coordinates 
    vector<LandmarkObs> transformed_observations;
    for(unsigned int j=0; j < observations.size(); j++) {
      double transformed_x = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
      double transformed_y = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
      transformed_observations.push_back(LandmarkObs{observations[j].id, transformed_x, transformed_y});
    }

    // assign predictions to landmarks
    dataAssociation(in_range, transformed_observations);

    // reset particle weights
    particles[i].weight = 1.0;

    for(unsigned int j=0; j < transformed_observations.size(); j++) {
      
      // observation and prediction coordinates
      double obs_x, obs_y, pre_x, pre_y;
      obs_x = transformed_observations[j].x;
      obs_y = transformed_observations[j].y;
      
      // get coordinates
      for(unsigned int k=0; k < in_range.size(); k++) {
        if(in_range[k].id == transformed_observations[j].id) {
          pre_x = in_range[k].x;
          pre_y = in_range[k].y;
        }
      }

      // calculate the weight
      double dx = std_landmark[0];
      double dy = std_landmark[1];
      double obs_weight = (1/(2*M_PI*dx*dy)) * exp(-(pow(pre_x-obs_x,2)/(2*pow(dx, 2)) + (pow(pre_y-obs_y,2)/(2*pow(dy, 2)))));

      // product of this obersvation weight with total observations weight
      particles[i].weight *= obs_weight;
    }
  }
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // get current weights
  vector<double> weights;
  for(int i=0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution from ranging from 0 to max_weight
  uniform_real_distribution<double> uni_real_dist(0.0, max_weight);

  // starting index
  uniform_int_distribution<int> uni_int_dist(0, num_particles-1);
  int index = uni_int_dist(gen);

  double beta = 0.0;

  vector<Particle> new_particles;

  // update particles
  for(int i=0; i < num_particles; i++) {
    beta += uni_real_dist(gen)*2.0;

    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
