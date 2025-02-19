https://ophtha-deployment.apps.wadhwaniai.org/optha-fastapi-service-8001/


POST /v2/models/pipeline/infer HTTP/1
Host: https://ophtha-deployment.apps.wadhwaniai.org/optha-inference-service-8000/
Content-Type: application/json
Content-Length: <xx>
{
  "id" : "42",
  "inputs" : [
    {
      "name" : "input0",
      "shape" : [ 2, 2 ],
      "datatype" : "UINT32",
      "data" : [ 1, 2, 3, 4 ]
    },
    {
      "name" : "input1",
      "shape" : [ 3 ],
      "datatype" : "BOOL",
      "data" : [ true ]
    }
  ],
  "outputs" : [
    {
      "name" : "output0"
    }
  ]
}