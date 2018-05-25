#include "application.h"
#include "net.h"

#include <boost/program_options.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

Application::Application(int argc, char* argv[])
    : trainingEpochs{1000}
    , verboseOutput{false}
    , logger{nullptr}
    , net{nullptr}
    , probingFreq{20}
    , loggerPath{}
{
    init(argc, argv);
}

void Application::init(int argc, char* argv[])
{
    double momentum = .9;
    double learnF = .1;
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

    logger = new Logger{loggerPath, verboseOutput, probingFreq};
    net = new Net{biasPresent, momentum, learnF,
                  layerConfiguration, rng, *logger};

    std::string arguments;
    for (size_t i = 0; i < argc; ++i) {
        arguments.append(argv[i]);
        arguments.append(" ");
    }
    logger->addToStream(arguments);

    auto [learn, test, validate] =
          processIris({"../data/iris.csv"}, rng);

    auto [inputLearn, outputLearn] = learn;
    auto [inputTest, outputTest] = test;
    auto [inputValidate, outputValidate] = validate;

    this->inputLearn = std::move(inputLearn);
    this->outputLearn = std::move(outputLearn);

    this->inputTest = std::move(inputTest);
    this->outputTest = std::move(outputTest);

    this->inputValidate = std::move(inputValidate);
    this->outputValidate = std::move(outputValidate);
}

Application::~Application()
{
    delete logger;
    delete net;
}

void Application::runNetwork(bool learning)
{
    // TODO refactor to runNetworkInternal und runNetwork
    if (learning) {
        size_t i = 0;
        logger->setItCounter(&i);
        for (; i < trainingEpochs; ++i) {
            if (logger->isVerbose()) {
                logger->addToStream({"Iteration " + std::to_string(i)});
            }

            auto timeToLearn = static_cast<bool>(i % probingFreq);

            auto& input  = timeToLearn ? inputLearn  : inputTest;
            auto& output = timeToLearn ? outputLearn : outputTest;

            std::vector<size_t> indexes(input.size());
            std::iota(std::begin(indexes), std::end(indexes), 0);

            if (likely(timeToLearn)) {
                std::shuffle(std::begin(indexes), std::end(indexes), rng);
            }

            double err = .0;

            for (size_t j = 0; j < indexes.size(); ++j) {
                err = net->run(input[indexes[j]],
                               output[indexes[j]],
                               timeToLearn);
            }

            logger->addToStream({std::to_string(i) +
            "," + std::to_string(err)});
        }
    } else {
        double err = .0;
        logger->addToStream({"Validation!"});
        for (size_t i = 0; i < inputValidate.size(); ++i) {
            err = net->run(inputValidate[i],
                           outputValidate[i],
                           false);
        }
        logger->addToStream({"Error: " + std::to_string(err)});
    }
}

