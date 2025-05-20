function updateVideo() {
    const model = document.getElementById('realtime_model').value;
    const videoFeed = document.getElementById('video_feed');
    videoFeed.src = `/video_feed/${model}`;
}

window.onload = function() {
    if (document.getElementById('realtime_model')) {
        updateVideo();
    }
};