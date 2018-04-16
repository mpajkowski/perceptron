#include "net.h"
#include "func.h"
#include "neuron.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <cstdlib>

Net::Net(int argc, char* argv[])
    : trainingEpochs{2200}
    , biasPresent{false}
    , momentum{.8}
    , learnF{.2}
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
            "specifies network layers configuration, i.e. 4 3 4")
        ("epochs,e", po::value<size_t>(&trainingEpochs),
            "specifies number of epochs for training, default: 2200")
        ("momentum,m", po::value<double>(&momentum),
            "specifies momentum factor, default: xD")
        ("learning-rate,l", po::value<double>(&learnF),
            "specifies learning rate factor, default: 0.02")
        ("with-bias,b", po::bool_switch(&biasPresent),
            "this option toggles on bias (1.)");

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
    populateLayers(layerConfiguration);
}

void Net::populateLayers(std::vector<size_t> const& layerConfiguration)
{
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
                                                 rng);
            }
        }
    }

    for (size_t i = 0; i < layerConfiguration.back(); ++i) {
        outputLayer.emplace_back(populateHidden ? hiddenLayers.back().size()
                                                : inputLayer.size(),
                                 rng);
    }
}

void Net::forwardPropagation()
{
    for (size_t i = 0; i < hiddenLayers[0].size(); ++i) {
        hiddenLayers[0][i].setInputs(std::move(std::vector<double>(inputLayer)));

        double sum = .0;
        for (size_t j = 0; j < hiddenLayers[0][i].inputs.size(); ++j) {
            sum += hiddenLayers[0][i].weights[j] * hiddenLayers[0][i].inputs[j];
        }
        if (biasPresent) sum += hiddenLayers[0][i].biasWeight;
        hiddenLayers[0][i].output = sigmoid::function(sum);
    }

    auto perLayerInputMatch = [](layer_t const& previousLayer)
        -> std::vector<double>
    {
        std::vector<double> previousLayerOutputs;
        for (size_t i = 0; i < previousLayer.size(); ++i) {
            previousLayerOutputs.push_back(previousLayer[i].output);
        }
        return previousLayerOutputs;
    };

    for (size_t i = 1; i < hiddenLayers.size(); ++i) {
        for (size_t j = 0; j < hiddenLayers[i].size(); ++j) {
            hiddenLayers[i][j].setInputs(
                    std::move(perLayerInputMatch(hiddenLayers[i - 1]))
            );

            double sum = .0;
            for (size_t k = 0; k < hiddenLayers[i][j].inputs.size(); ++k) {
               sum += hiddenLayers[i][j].inputs[k] * hiddenLayers[i][j].weights[k];
            }
            if (biasPresent) sum += hiddenLayers[i][j].biasWeight;
            hiddenLayers[i][j].output = sigmoid::function(sum);
        }
    }

    for (size_t i = 0; i < outputLayer.size(); ++i) {
        outputLayer[i].setInputs(
                std::move(perLayerInputMatch(hiddenLayers.back()))
        );

        double sum = .0;
        for (size_t j = 0; j < outputLayer[i].inputs.size(); ++j) {
            sum += outputLayer[i].inputs[j] * outputLayer[i].weights[j];
        }
        if (biasPresent) sum += outputLayer[i].biasWeight;
        outputLayer[i].output = sigmoid::function(sum);
    }
}

void Net::backPropagation(std::vector<double> const& trainingSet)
{
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        auto& neuron = outputLayer[i];

        double sum = .0;
        for (size_t j = 0; j < neuron.inputs.size(); ++j) {
            sum += neuron.inputs[j] * neuron.weights[j];
        }
        if (biasPresent) sum += neuron.biasWeight;

        neuron.error = (trainingSet[i] - sigmoid::function(sum))
          * sigmoid::derivative(sum);
    }

    size_t i = hiddenLayers.size() - 1;
    for (size_t j = 0; j < hiddenLayers[i].size(); j++) {
        for (size_t k = 0; k < outputLayer.size(); k++) {
            hiddenLayers[i][j].error += outputLayer[k].error * outputLayer[k].weights[j];
        }
        hiddenLayers[i][j].error *= sigmoid::derivative(hiddenLayers[i][j].output);
    }

    while (i --> 0) {
        for (size_t j = 0; j < hiddenLayers[i].size(); j++) {
            for (size_t k = 0; k < hiddenLayers[i + 1].size(); k++) {
                hiddenLayers[i][j].error += hiddenLayers[i + 1][k].error * hiddenLayers[i + 1][k].weights[j];
            }
            hiddenLayers[i][j].error *= sigmoid::derivative(hiddenLayers[i][j].output);
        }
    }
}

void Net::updateNeurons()
{
    for (auto& layer : hiddenLayers) {
        for (auto& neuron : layer) {
            neuron.update(momentum, learnF, biasPresent);
        }
    }

    for (auto& neuron : outputLayer) {
        neuron.update(momentum, learnF, biasPresent);
    }
}

void Net::training()
{
    auto regularTest = [this]
    {
        auto t = test();
        std::cout << "Good: " << t.first << ", Bad: " << t.second << "\n";
    };

    auto [inputLearnSignals, outputLearnSignals] =
          createDataset<8>(std::string{"../data/assign_in2out.csv"}, 4, 4, 4);

    auto shuffleIndexes = [this](size_t setSize)
    {
        std::vector<size_t> indexes(setSize);
        std::generate(indexes.begin(), indexes.end(),
                [this, &setSize] { return --setSize; });
        std::shuffle(indexes.begin(), indexes.end(), rng);
        return indexes;
    };

    auto shuffledIndexes = shuffleIndexes(inputLearnSignals.size());

    while (trainingEpochs --> 0) {
        for (size_t i = 0; i < shuffledIndexes.size(); ++i) {
            inputLayer = inputLearnSignals[shuffledIndexes[i]];
            forwardPropagation();
            backPropagation(outputLearnSignals[shuffledIndexes[i]]);
            updateNeurons();
        }

        if (!(trainingEpochs % 1000)) {
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

    auto [inputTestSignals, outputTestSignals] =
        createDataset<8>(std::string{"../data/assign_in2out.csv"}, 4, 4, 4);

    for (size_t i = 0; i < inputTestSignals.size(); ++i) {
        inputLayer = inputTestSignals[i];
        forwardPropagation();

        int positiveInRow = 0;
        int negativeInRow = 0;

        std::cout << "-----\n";
        for (size_t j = 0; j < outputLayer.size(); ++j) {

            double wanted = outputTestSignals[i][j];
            double response = outputLayer[j].output;
            double error = wanted - response;

            if (error * error < 0.05) {
                ++positiveInRow;
            }
            else
            {
                ++negativeInRow;
            }
            std::cout << "wanted: " << wanted << ", response: " << response << std::endl;
        }
        if (positiveInRow == outputLayer.size()) {
            ++positiveAnswers;
        } else {
            ++negativeAnswers;
        }
    }

    return std::make_pair(positiveAnswers, negativeAnswers);
}

