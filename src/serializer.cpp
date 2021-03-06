#include "serializer.h"
#include "application.h"
#include "net.h"
#include "neuron.h"

using namespace tinyxml2;

Serializer::Serializer(Application* application)
    : application{application}
{}

void Serializer::saveData()
{
    XMLDocument xmlDoc;
    auto net = application->net;
    auto _network = xmlDoc.NewElement("Network");
    xmlDoc.InsertFirstChild(_network);

    auto _config = xmlDoc.NewElement("Config");
    auto _biasPresent = xmlDoc.NewElement("BiasPresent");
    _biasPresent->SetText(net->biasPresent);
    _config->InsertEndChild(_biasPresent);
    auto _momentum = xmlDoc.NewElement("Momentum");
    _momentum->SetText(net->momentum);
    _config->InsertEndChild(_momentum);
    auto _learnF = xmlDoc.NewElement("LearnF");
    _learnF->SetText(net->learnF);
    _config->InsertEndChild(_learnF);
    _network->InsertEndChild(_config);

    auto _layers = xmlDoc.NewElement("HiddenLayers");
    for (auto const& layer : net->layers) {
        auto _layer = xmlDoc.NewElement("Layer");
        for (auto const& neuron : layer) {
            auto _neuron = xmlDoc.NewElement("Neuron");

            auto _biasWeight = xmlDoc.NewElement("BiasWeight");
            _biasWeight->SetText(neuron.biasWeight);
            _neuron->InsertEndChild(_biasWeight);

            auto _biasPWeight = xmlDoc.NewElement("BiasPWeight");
            _biasPWeight->SetText(neuron.biasPWeight);
            _neuron->InsertEndChild(_biasPWeight);

            auto _error = xmlDoc.NewElement("Error");
            _error->SetText(neuron.error);
            _neuron->InsertEndChild(_error);

            auto _output = xmlDoc.NewElement("Output");
            _output->SetText(neuron.output);
            _neuron->InsertEndChild(_output);

            auto _weights = xmlDoc.NewElement("Weights");
            for (auto const& weight : neuron.weights) {
                auto _weight = xmlDoc.NewElement("Weight");
                _weight->SetText(weight);
                _weights->InsertEndChild(_weight);
            }
            _neuron->InsertEndChild(_weights);

            auto _pWeights = xmlDoc.NewElement("PreviousWeights");
            for (auto const& pWeight : neuron.pWeights) {
                auto _pWeight = xmlDoc.NewElement("PreviousWeight");
                _pWeight->SetText(pWeight);
                _pWeights->InsertEndChild(_pWeight);
            }
            _neuron->InsertEndChild(_pWeights);
            _layer->InsertEndChild(_neuron);
        }
        _layers->InsertEndChild(_layer);
    }
    _network->InsertEndChild(_layers);

    if (application->serializerPath != "") {
        xmlDoc.SaveFile(application->serializerPath.c_str());
    }
}

