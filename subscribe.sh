# Command to subscribe to make a running container subscribe to all the entities from which it should receive notifications


# By default we update the subscription, so first we delete it if it existed
curl -X DELETE 'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/subscriptions/urn:ngsi-ld:fhhi:one_subscription_to_rule_them_all'



# define the POST_DATA_URL
POST_DATA_URL="http://<your_ip_address>:<PORT>/tfa02/post_data"

echo "POST_DATA_URL: '$POST_DATA_URL'"

# PersonVehicleDetection subscription
curl -X POST 'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/subscriptions' \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "urn:ngsi-ld:fhhi:one_subscription_to_rule_them_all",
    "type": "Subscription",
    "name": "TFA02 inputs subscription",
    "description": "Subscription description",
    "entities": [
        {
            "type": "EOFloodExtent",
        },  
        {
            "type": "EOBurntArea",
        },
        {
            "type": "BurntSegmentation"
        },
        {
            "type": "FireSegmentation"
        },
        {
            "type": "FloodSegmentation"
        },
        {
            "type": "PersonVehicleDetection"
        },
        {
            "type": "SmokeSegmentation"
        }
    ],
    "notification": {
        "endpoint": {
            "uri": "$POST_DATA_URL", 
            "accept": "application/json"
        }
    }
}'


