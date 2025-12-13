//<script>
//    document.addEventListener('DOMContentLoaded', function() {
//        const diagnoseForm = document.getElementById('diagnose-form');
//        const diagnoseButton = document.getElementById('diagnoseButton');
//        const imageUpload = document.getElementById('imageUpload');
//        const imagePreview = document.getElementById('imagePreview');
//        const resultContainer = document.getElementById('resultContainer');
//        const predictedClassSpan = document.getElementById('predictedClass');
//        const confidenceScoreSpan = document.getElementById('confidenceScore');
//
//        // Hàm xem trước ảnh
//        imageUpload.addEventListener('change', function() {
//            if (this.files && this.files[0]) {
//                const reader = new FileReader();
//                reader.onload = function(e) {
//                    imagePreview.innerHTML = `<img src="${e.target.result}" alt="Xem trước ảnh" style="max-width: 100%; max-height: 150px; object-fit: contain;">`;
//                    diagnoseButton.disabled = false; // Bật nút chẩn đoán
//                }
//                reader.readAsDataURL(this.files[0]);
//            } else {
//                imagePreview.innerHTML = '<p class="text-muted">Xem trước ảnh tải lên</p>';
//                diagnoseButton.disabled = true;
//            }
//        });
//
//        // Hàm xử lý khi nhấn nút Chẩn đoán
//        diagnoseButton.addEventListener('click', async function() {
//            if (imageUpload.files.length === 0) {
//                alert("Vui lòng chọn một ảnh để chẩn đoán.");
//                return;
//            }
//
//            const formData = new FormData(diagnoseForm);
//
//            // Hiển thị trạng thái đang tải
//            diagnoseButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang chẩn đoán...';
//            diagnoseButton.disabled = true;
//
//            try {
//                const response = await fetch('/diagnose', {
//                    method: 'POST',
//                    body: formData
//                });
//
//                const result = await response.json();
//
//                if (response.ok) {
//                    // Cập nhật kết quả
//                    predictedClassSpan.textContent = result.predicted_class;
//                    confidenceScoreSpan.textContent = result.confidence;
//                    resultContainer.classList.remove('d-none');
//                } else {
//                    // Xử lý lỗi từ server
//                    alert('Lỗi chẩn đoán: ' + result.error);
//                }
//
//            } catch (error) {
//                console.error('Lỗi khi gửi yêu cầu:', error);
//                alert('Đã xảy ra lỗi kết nối. Vui lòng thử lại.');
//            } finally {
//                // Khôi phục nút
//                diagnoseButton.innerHTML = 'Chẩn đoán (AI)';
//                diagnoseButton.disabled = false;
//            }
//        });
//    });
//</script>