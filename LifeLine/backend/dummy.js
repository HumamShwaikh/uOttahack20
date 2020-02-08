$.ajax({
    type: "POST",
    url: "~/server.py",
    data: { param: text}
  }).done(function( o ) {
    console.log(data);
});