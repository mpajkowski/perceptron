#include "application.h"
#include "net.h"
#include "func.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

Application::Application(int argc, char* argv[])
    : trainingEpochs{1000}
    , verboseOutput{false}
    , serializerPath{""}
    , loggerPath{""}
    , fileLogger{nullptr}
{
    init(argc, argv);
}

void Application::init(int argc, char* argv[])
{
    double momentum = .0;
    double learnF = .9;
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
            "specifies momentum factor, default: 0.0")
        ("learning-rate,l", po::value<double>(&learnF),
            "specifies learning rate factor, default: 0.9")
        ("with-bias,b", po::bool_switch(&biasPresent),
            "this option toggles on bias (1. input for each neuron)")
        ("verbose,v", po::bool_switch(&verboseOutput),
            "toggles on verbose output")
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
    if (loggerPath != "") {
        fileLogger = new FileLogger{loggerPath};
    }
}

Application::~Application()
{
    delete fileLogger;
    delete net;
}

void Application::runNetwork(bool train)
{
    auto [inputSignals, outputSignals] =
          createDataset<8>(std::string{"../data/assign_in2out.csv"}, 4, 4, 4);

    std::vector<size_t> indexes(inputSignals.size());
    std::iota(std::begin(indexes), std::end(indexes), 0);

    size_t numIterations = train ? trainingEpochs : 1;
    for (size_t i = 0; i < numIterations; ++i) {
        double err = .0;

        if (train) {
            std::shuffle(std::begin(indexes), std::end(indexes), rng);
        }

        for (size_t j = 0; j < indexes.size(); ++j) {
            err = net->run(inputSignals[indexes[j]],
                           outputSignals[indexes[j]],
                           train);

            if (i % 20 == 0) {
                std::string output{std::to_string(i) +
                    "," + std::string{std::to_string(err)}};
                if (fileLogger) {
                    fileLogger->addToStream(output);
                } else {
                    std::cout << output << "\n";
                }
            }

            if (__builtin_expect(!train, 0)) {
               if (fileLogger) {
                    fileLogger->addToStream(std::to_string(err));
                } else {
                    std::cout << err << "\n";
                }
            }
        }
    }
}

