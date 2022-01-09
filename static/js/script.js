var api_url = "/api/pred/";

var models = ["alexnet", "resnet", "squeezenet", "vgg", "densenet", "googlenet", "shufflenet", "mobilenet", "resnext", "wide_resnet", "mnasnet", "efficientnet", "regnet_x", "regnet_y"];

function apiReq() {

    var img_url = document.getElementById('image_url').value;

    var xhr = new Array(models.length);    
 
    for (i = 0; i < models.length; i++)
    {
        var nn_model = models[i];

        xhr[i] = new XMLHttpRequest();
        xhr[i].open("POST", api_url, true);
        xhr[i].setRequestHeader("Content-Type", "application/json");
        
        var data = JSON.stringify({"url": img_url, "model": nn_model});
        xhr[i].send(data);
    }
};
