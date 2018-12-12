import js2py
import requests
# js = """
# function escramble_758(){
# 	var url = "https://api.nytimes.com/svc/archive/v1/2015/1.json";
# 	url += '?' + $.param({
# 	  'api-key': "c83a82484d594c20bc0568e4f183f0f3"
# 	});
# 	$.ajax({
# 	  url: url,
# 	  method: 'GET',
# 	}).done(function(result) {
# 	  console.log(result);
# 	}).fail(function(err) {
# 	  throw err;
# 	});

# }
# escramble_758()
# """.replace("document.write", "return ")

# result = js2py.eval_js(js)  # executing JavaScript and converting the result to python string 
# print(result)


parameters = {'api-key': "c83a82484d594c20bc0568e4f183f0f3"}

response = requests.get("https://api.nytimes.com/svc/archive/v1/2015/1.json", params=parameters)
json = response.json()
print(json['response']['meta'])