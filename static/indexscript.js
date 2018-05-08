var jumpToGraph = function() {
  console.log("hello");
  window.location.href = "#results";

  var body = document.getElementById("main");
  body.className += ' fade-in';
  body.style.opacity = 1;
}

var fadeIn = function() {
  var body = document.getElementById("main");
  body.className += ' fade-in';
  body.style.opacity = 1;
}
