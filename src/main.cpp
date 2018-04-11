#include "propagation.h"
#include "func.h"

#include <random>
#include <iostream>
#include <boost/program_options.hpp>

int main(int argc, char* argv[])
{
    namespace po = boost::program_options;
    std::vector<size_t> layerConfiguration;

    po::options_description desc("Options:");
    desc.add_options()
        ("help,h", "prints this help message")
        ("configuration,c", po::value<std::vector<size_t>>()->multitoken()->required(),
             "specifies network configuration, i.e. 4 3 4")
        ("with-bias,b", "this option toggles on bias input for every neuron in network");

    po::variables_map vm;
////// TODO FIX
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (!vm["configuration"].empty()) {
            layerConfiguration = vm["configuration"].as<std::vector<size_t>>();
        }
    }
    catch (po::error& e) {
        std::cerr << "Error " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    std::vector<double> inputLayer;
    std::vector<layer_t> hiddenLayers;
    layer_t outputLayer;

    std::vector<double> h1;
    std::vector<double> h2;

    std::mt19937 rng;
    rng.seed(std::random_device{}());

    // Populate layers
    for (size_t i = 0; i < layerConfiguration[0]; ++i) {
        inputLayer.emplace_back();
    }

    bool populateHidden = layerConfiguration.size() > 2;
    
    if (populateHidden) {
        for (size_t i = 1; i < layerConfiguration.size() - 1; ++i) {
            hiddenLayers.emplace_back();
            for (size_t j = 0; j < layerConfiguration[i]; ++j) {
                hiddenLayers[i - 1].emplace_back(i == 1 ? inputLayer.size()
                                                        : hiddenLayers[i - 2].size(),
                                                 true,
                                                 rng);
            }
        }
    }

    for (size_t i = 0; i < layerConfiguration.back(); ++i) {
        outputLayer.emplace_back(populateHidden ? hiddenLayers.back().size()
                                                : inputLayer.size(),
                                 true,
                                 rng);
    }
    
    adjustHelpers(inputLayer, hiddenLayers, outputLayer, h1, h2);
    training(2200, inputLayer, hiddenLayers, outputLayer, h1, h2, rng);
}
