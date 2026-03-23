curl -X POST 'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/subscriptions' \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "urn:ngsi-ld:fhhi:test_subscription",
    "type": "Subscription",
    "name": "Subscription name",
    "description": "Subscription description",
    "entities": [
        {
            "type": "PersonVehicleDetectionTest"
        },
        {
            "type": "PersonVehicleDetectionExplanation"
        }
    ],
    "notification": {
        "endpoint": {
            "uri": "https://webhook.site/7804e6ce-74a8-4730-aefb-f96df450daf2",
            "accept": "application/json"
        }
    }
}'