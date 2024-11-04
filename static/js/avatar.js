let scene, camera, renderer, avatar;

function initAvatar() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / 2 / window.innerHeight, 0.1, 1000);
    
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth / 2, window.innerHeight);
    document.getElementById('avatar-container').appendChild(renderer.domElement);

    // Load SMPL-X model
    const loader = new THREE.ObjectLoader();
    loader.load('/static/models/avatar_model.json', function(obj) {
        avatar = obj;
        scene.add(avatar);
    });

    camera.position.z = 5;
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

function updateAvatarPose(poseData) {
    if (!avatar) return;
    
    // Map pose landmarks to avatar joints
    // This is a simplified version - actual implementation would need full SMPL-X integration
    avatar.rotation.x = poseData[0].y;
    avatar.rotation.y = poseData[0].x;
}

window.addEventListener('load', initAvatar);
