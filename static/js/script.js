var api_url = "/api/pred/";

var models = ["alexnet", "resnet", "squeezenet", "vgg", "densenet", "googlenet", "shufflenet", "mobilenet", "resnext", "wide_resnet", "mnasnet", "efficientnet", "regnet_x", "regnet_y"];

var xhr = new Array(models.length);

function checkImage() {

    var url = document.getElementById('image_url').value;

    var image = new Image();
    image.onload = function() {
        if (this.width > 0) {
          apiReq(url);
        }
    }
    image.onerror = function() {
        alert("Please enter a valid imag url!!");;
    }
    image.src = url;
}

function apiReq(img_url) {
 
    for (i = 0; i < models.length; i++)
    {
        var nn_model = models[i];

        xhr[i] = new XMLHttpRequest();
        xhr[i].open("POST", api_url, true);
        xhr[i].setRequestHeader("Content-Type", "application/json");
        
        var data = JSON.stringify({"url": img_url, "model": nn_model});
        xhr[i].onreadystatechange = function () {
            if (this.readyState === 4 && this.status === 200) {

                var data = JSON.parse(this.response)

                if (document.getElementById('form') != null){
                    document.getElementById('form').remove();
                }

                if (document.getElementsByClassName('image-view')[0] == null){
                    document.getElementsByClassName('box')[0].innerHTML = "<div class='image-view'> <img src=" + img_url+ " alt='Input Image'> </div>";
                    document.getElementsByClassName('box')[0].style.display="block";
                }

                if (document.getElementsByClassName('pred-detail')[0] == null){
                    var predDetailDiv = document.createElement('div');
                    predDetailDiv.className = "pred-detail";
                    document.getElementsByClassName('box')[0].appendChild(predDetailDiv);
               }

                var table = document.createElement('table');
                table.innerHTML= "<tr><th colspan='2' style='color: yellowgreen; text-transform: capitalize;'>" + data.nnModel + "</th></tr><tr><th>Prediction</th><th>Confidence Percentage</th></tr><tr><td>"+ data.clsName1 +"</td><td>" + data.percent1 +"</td></tr><tr><td>"+ data.clsName2 +"</td><td>"+ data.percent2 +"</td></tr><tr><td>"+ data.clsName3 +"</td><td>"+ data.percent3 +"</td></tr>";
                document.getElementsByClassName('pred-detail')[0].appendChild(table);

                if (document.getElementsByClassName('home-link')[0] == null){
                    var homeLinkDiv = document.createElement('div');
                    homeLinkDiv.className = "home-link";
                    homeLinkDiv.innerHTML = "<button class='submit-btn'><a href='/'>Home</a></button>";
                    document.getElementsByClassName('box')[0].appendChild(homeLinkDiv);
                    document.getElementsByClassName('submit-btn')[0].style.marginLeft="unset";
                }

            }
        };
        xhr[i].send(data);

    }

};
