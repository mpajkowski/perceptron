#include "application.h"
#include "net.h"
#include "func.h"

#include <boost/program_options.hpp>
#include <cstdlib>
#include <cmath>
#include <iostream>

Application::Application(int argc, char* argv[])
    : trainingEpochs{1000}
    , verboseOutput{false}
    , serializerPath{""}
    , loggerPath{""}
    , logger{nullptr}
    , probingFreq{20}
{
    init(argc, argv);
}

void Application::init(int argc, char* argv[])
{
    double momentum = .9;
    double learnF = .2;
    bool biasPresent = false;

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
            "specifies momentum factor, default: 0.9")
        ("learning-rate,l", po::value<double>(&learnF),
            "specifies learning rate factor, default: 0.2")
        ("with-bias,b", po::bool_switch(&biasPresent),
            "this option toggles on bias (1. input for each neuron)")
        ("verbose,v", po::bool_switch(&verboseOutput),
            "toggles on verbose output")
        ("probing-freq,f", po::value<size_t>(&probingFreq),
            "sets probing frequency value, default: 20")
        ("serialize,s", po::value<std::string>(&serializerPath),
            "serialize to XML")
        ("log-learning", po::value<std::string>(&loggerPath),
            "log learning results to file, format (epochs,mse)");

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (!vm["configuration"].empty()) {
            layerConfiguration = vm["configuration"].as<std::vector<size_t>>();
        }
    } catch (po::error& e) {
        std::cerr << "Error " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        exit(1);
    }

    rng.seed(std::random_device{}());

    net = new Net{biasPresent, momentum, learnF,
                  layerConfiguration, rng};
    logger = new Logger{loggerPath};
}

Application::~Application()
{
    delete logger;
    delete net;
}

void Application::runNetwork(bool train)
{
    auto [inputLearn, outputLearn, inputTest, outputTest] =
          processIris(std::string{"../data/iris.csv"});

    auto& inputSignals  = train ? inputLearn  : inputTest;
    auto& outputSignals = train ? outputLearn : outputTest;

    std::vector<size_t> indexes(inputSignals.size());
    std::iota(std::begin(indexes), std::end(indexes), 0);

    size_t numIterations = train ? trainingEpochs : 1;
    for (size_t i = 0; i < numIterations; ++i) {
        double err = .0;

        if (likely(train)) {
            std::shuffle(std::begin(indexes), std::end(indexes), rng);
        }

        for (size_t j = 0; j < indexes.size(); ++j) {
            err = net->run(inputSignals[indexes[j]],
                           outputSignals[indexes[j]],
                           train);
        }

        if (i % probingFreq == 0) {
            std::string output;
            output.append(std::to_string(i) +
                "," + std::string{std::to_string(err)});
            logger->addToStream(output);
        }
    }
}

