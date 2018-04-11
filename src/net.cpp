#include "net.h"
#include "func.h"
#include "neuron.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <cstdlib>

Net::Net(int argc, char* argv[])
{
    init(argc, argv);
}

void Net::init(int argc, char* argv[])
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
        exit(1);
    }

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
    
    adjustHelpers();
}

void Net::adjustHelpers()
{
    std::vector<size_t> s;

    s.push_back(inputLayer.size());
    s.push_back(outputLayer.size());

    for (auto& layer : hiddenLayers) {
        s.push_back(layer.size());
    }

    auto maxVal = std::max_element(s.begin(), s.end());
    _1.resize(*maxVal);
    _2.resize(*maxVal);
    std::fill(_1.begin(), _1.end(), .0);
    std::fill(_2.begin(), _2.end(), .0);
}

void Net::training(size_t epochs)
{
    auto regularTest = [&]() 
    {
        auto t = test();
        std::cout << "Good: " << t.first << ", Bad: " << t.second << "\n";
    };
   
    auto [inputLearnSignals, outputLearnSignals] = createDataset(0., 100., 1000,
            inputLayer.size(), rng, [](double x) { return std::sqrt(x); });

    while (epochs --> 0) {
        for (size_t i = 0; i < inputLearnSignals.size(); ++i) {
            inputLayer = inputLearnSignals[i];
            _1 = inputLayer;

            forwardPropagation();
            backPropagation(outputLearnSignals[i]);
            updateNeurons();
        }

        if (!(epochs % 1000)) {
            for (auto& layer : hiddenLayers) {
                for (auto& neuron : layer) {
                    neuron.learnF += .001;
                }
            }
            regularTest();
        }
    }
    std::cout << "Learning ended!\n";
    regularTest();
    regularTest();
    regularTest();
}

std::pair<int, int> Net::test()
{
    int positiveAnswers = 0;
    int negativeAnswers = 0;

    auto [inputTestSignals, outputTestSignals] = createDataset(0., 100., 1000,
            inputLayer.size(), rng, [](double x) { return std::sqrt(x); });

    for (size_t i = 0; i < inputTestSignals.size(); ++i) {
        inputLayer = inputTestSignals[i];
        _1 = inputLayer;
        forwardPropagation();

        for (size_t j = 0; j < outputLayer.size(); ++j) {
            outputLayer[j].setInputs(_2);
            outputLayer[j].updateSum();

            double estimator = outputTestSignals[i][j];
            double response = sigmoid::function(outputLayer[j].sum) * 10;
            double error = estimator - response;

            if (std::abs(error) < 0.5) {
                ++positiveAnswers;
            }
            else
            {
                ++negativeAnswers;
            }
        }
    }

    return std::make_pair(positiveAnswers, negativeAnswers);
}

void Net::forwardPropagation()
{
    for (auto& layer : hiddenLayers) {
        for (size_t i = 0; i < layer.size(); ++i) {
            layer[i].setInputs(_1);
            layer[i].updateSum();
            _2[i] = sigmoid::function(layer[i].sum);
        }
        _1 = _2;
    }
}

void Net::backPropagation(std::vector<double> const& trainingSet)
{
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        outputLayer[i].setInputs(_2);
        outputLayer[i].updateSum();
        outputLayer[i].delta = ((trainingSet[i] / 10) - sigmoid::function(outputLayer[i].sum))
          * sigmoid::derivative(outputLayer[i].sum);
    }

    size_t i = hiddenLayers.size() - 1;
    for (size_t j = 0; j < hiddenLayers[i].size(); j++) {
        for (size_t k = 0; k < outputLayer.size(); k++) {
            hiddenLayers[i][j].delta += outputLayer[k].delta * outputLayer[k].weights[j];
        }
        hiddenLayers[i][j].delta *= sigmoid::derivative(hiddenLayers[i][j].sum);
    }

    while (i --> 0) {
        for (size_t j = 0;j<hiddenLayers[i].size();j++) {
            for (size_t k = 0; k < hiddenLayers[i + 1].size(); k++) {
                hiddenLayers[i][j].delta += hiddenLayers[i + 1][k].delta * hiddenLayers[i + 1][k].weights[j];
            }
            hiddenLayers[i][j].delta *= sigmoid::derivative(hiddenLayers[i][j].sum);
        }
    }
}

void Net::updateNeurons()
{
    for (auto& layer : hiddenLayers) {
        for (auto& neuron : layer) {
            neuron.update();
        }
    }

    for (auto& neuron : outputLayer) {
        neuron.update();
    }
}
