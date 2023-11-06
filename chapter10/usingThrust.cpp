#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>

#include <vector>
#include <iostream>

int main(void) {
	// the following code shows how to use thrust::sort and thrust::host_vector
	std::vector<int> array = {2, 4, 6, 8, 0, 9, 7, 5, 3, 1};
	thrust::host_vector<int> vec;
	vec = array; 	// now vec has storage for 10 integers
	std::cout << "vec has size: " << vec.size() << std::endl;

	std::cout << "vec before sorting:" << std::endl;
	for (size_t i = 0; i < vec.size(); ++i)
	std::cout << vec[i] << "  ";
	std::cout << std::endl;

	thrust::sort(vec.begin(), vec.end());
	std::cout << "vec after sorting:" << std::endl;
	for (size_t i = 0; i < vec.size(); ++i)
			std::cout << vec[i] << "  ";
	std::cout << std::endl;

	vec.resize(2);
	std::cout << "now vec has size: " << vec.size() << std::endl;

	return 0;
}
