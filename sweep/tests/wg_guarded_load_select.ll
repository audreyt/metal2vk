; Self-check input for sweep/tests/run.py: clspv's optimiser speculates a guarded load from a
; Workgroup array into a select over two differently-typed workgroup pointers, and the producer
; emits an ill-typed OpSelect for it (spirv-val --target-env vulkan1.3:
; "Expected both objects to be of Result Type: Select"; reduced from the nvfp4 argmax kernel,
; sweep/mlx-batch/vlmb/repro/). guarded_wg_load_select.py must hoist the guarded addrspace(3)
; load into the branch block and turn the join phis into value selects before clspv runs.
; ModuleID = '/tmp/m2v-v20/ll/v19_v20_metal__v20_single_token_d9b7ace9.ll'
source_filename = "/tmp/m2v-v20/src/v19_v20_metal__v20_single_token_d9b7ace9.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

@_ZZ16v20_single_tokenE13group_indices = internal addrspace(3) global [4 x i32] undef, align 4

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local spir_kernel void @v20_single_token(ptr addrspace(1) noundef align 2 %0, ptr addrspace(1) noundef align 2 %1, ptr addrspace(1) noundef align 4 %2) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 !kernel_arg_name !9 {
  %4 = alloca [4 x float], align 4
  %5 = call spir_func i32 @_Z12get_group_idj(i32 noundef 0) #4
  %6 = call spir_func i32 @_Z12get_group_idj(i32 noundef 1) #4
  %7 = call spir_func i32 @_Z12get_group_idj(i32 noundef 2) #4
  %8 = insertelement <3 x i32> poison, i32 %5, i32 0
  %9 = insertelement <3 x i32> %8, i32 %6, i32 1
  %10 = insertelement <3 x i32> %9, i32 %7, i32 2
  %11 = shufflevector <3 x i32> %10, <3 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = call spir_func i32 @_Z13get_global_idj(i32 noundef 0) #4
  %13 = call spir_func i32 @_Z13get_global_idj(i32 noundef 1) #4
  %14 = call spir_func i32 @_Z13get_global_idj(i32 noundef 2) #4
  %15 = insertelement <3 x i32> poison, i32 %12, i32 0
  %16 = insertelement <3 x i32> %15, i32 %13, i32 1
  %17 = insertelement <3 x i32> %16, i32 %14, i32 2
  %18 = call spir_func i32 @_Z16get_sub_group_idv() #5
  %19 = call spir_func i32 @_Z22get_sub_group_local_idv() #5
  %20 = uitofp i32 %18 to float
  %21 = getelementptr inbounds nuw [4 x float], ptr %4, i32 0, i32 %18
  store float %20, ptr %21, align 4
  %22 = getelementptr inbounds nuw [4 x i32], ptr addrspace(3) @_ZZ16v20_single_tokenE13group_indices, i32 0, i32 %18
  store i32 %18, ptr addrspace(3) %22, align 4
  call spir_func void @_Z7barrierj(i32 noundef 1) #5
  %23 = icmp eq i32 %18, 0
  %24 = icmp eq i32 %19, 0
  %or.cond = select i1 %23, i1 %24, i1 false
  br i1 %or.cond, label %25, label %43

25:                                               ; preds = %3
  %26 = load float, ptr %4, align 4
  %27 = load i32, ptr addrspace(3) @_ZZ16v20_single_tokenE13group_indices, align 4
  br label %28

28:                                               ; preds = %37, %25
  %.031 = phi i32 [ 1, %25 ], [ %38, %37 ]
  %.029 = phi i32 [ %27, %25 ], [ %.130, %37 ]
  %.0 = phi float [ %26, %25 ], [ %.1, %37 ]
  %29 = icmp slt i32 %.031, 2
  br i1 %29, label %30, label %39

30:                                               ; preds = %28
  %31 = getelementptr inbounds [4 x float], ptr %4, i32 0, i32 %.031
  %32 = load float, ptr %31, align 4
  %33 = fcmp ogt float %32, %.0
  br i1 %33, label %34, label %37

34:                                               ; preds = %30
  %35 = getelementptr inbounds [4 x i32], ptr addrspace(3) @_ZZ16v20_single_tokenE13group_indices, i32 0, i32 %.031
  %36 = load i32, ptr addrspace(3) %35, align 4
  br label %37

37:                                               ; preds = %34, %30
  %.130 = phi i32 [ %36, %34 ], [ %.029, %30 ]
  %.1 = phi float [ %32, %34 ], [ %.0, %30 ]
  %38 = add nsw i32 %.031, 1
  br label %28, !llvm.loop !10

39:                                               ; preds = %28
  %40 = fptrunc float %.0 to half
  %41 = getelementptr inbounds half, ptr addrspace(1) %1, i32 %6
  store half %40, ptr addrspace(1) %41, align 2
  %42 = getelementptr inbounds i32, ptr addrspace(1) %2, i32 %6
  store i32 %.029, ptr addrspace(1) %42, align 4
  br label %43

43:                                               ; preds = %39, %3
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define dso_local spir_func void @__clang_ocl_kern_imp_v20_single_token(ptr addrspace(1) noundef align 2 %0, ptr addrspace(1) noundef align 2 %1, ptr addrspace(1) noundef align 4 %2) #1 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 !kernel_arg_name !9 {
  %4 = alloca [4 x float], align 4
  %5 = call spir_func i32 @_Z12get_group_idj(i32 noundef 0) #4
  %6 = call spir_func i32 @_Z12get_group_idj(i32 noundef 1) #4
  %7 = call spir_func i32 @_Z12get_group_idj(i32 noundef 2) #4
  %8 = insertelement <3 x i32> poison, i32 %5, i32 0
  %9 = insertelement <3 x i32> %8, i32 %6, i32 1
  %10 = insertelement <3 x i32> %9, i32 %7, i32 2
  %11 = shufflevector <3 x i32> %10, <3 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = call spir_func i32 @_Z13get_global_idj(i32 noundef 0) #4
  %13 = call spir_func i32 @_Z13get_global_idj(i32 noundef 1) #4
  %14 = call spir_func i32 @_Z13get_global_idj(i32 noundef 2) #4
  %15 = insertelement <3 x i32> poison, i32 %12, i32 0
  %16 = insertelement <3 x i32> %15, i32 %13, i32 1
  %17 = insertelement <3 x i32> %16, i32 %14, i32 2
  %18 = call spir_func i32 @_Z16get_sub_group_idv() #5
  %19 = call spir_func i32 @_Z22get_sub_group_local_idv() #5
  %20 = uitofp i32 %18 to float
  %21 = getelementptr inbounds nuw [4 x float], ptr %4, i32 0, i32 %18
  store float %20, ptr %21, align 4
  %22 = getelementptr inbounds nuw [4 x i32], ptr addrspace(3) @_ZZ16v20_single_tokenE13group_indices, i32 0, i32 %18
  store i32 %18, ptr addrspace(3) %22, align 4
  call spir_func void @_Z7barrierj(i32 noundef 1) #5
  %23 = icmp eq i32 %18, 0
  %24 = icmp eq i32 %19, 0
  %or.cond = select i1 %23, i1 %24, i1 false
  br i1 %or.cond, label %25, label %43

25:                                               ; preds = %3
  %26 = load float, ptr %4, align 4
  %27 = load i32, ptr addrspace(3) @_ZZ16v20_single_tokenE13group_indices, align 4
  br label %28

28:                                               ; preds = %37, %25
  %.027 = phi i32 [ %27, %25 ], [ %.128, %37 ]
  %.026 = phi i32 [ 1, %25 ], [ %38, %37 ]
  %.0 = phi float [ %26, %25 ], [ %.1, %37 ]
  %29 = icmp slt i32 %.026, 2
  br i1 %29, label %30, label %39

30:                                               ; preds = %28
  %31 = getelementptr inbounds [4 x float], ptr %4, i32 0, i32 %.026
  %32 = load float, ptr %31, align 4
  %33 = fcmp ogt float %32, %.0
  br i1 %33, label %34, label %37

34:                                               ; preds = %30
  %35 = getelementptr inbounds [4 x i32], ptr addrspace(3) @_ZZ16v20_single_tokenE13group_indices, i32 0, i32 %.026
  %36 = load i32, ptr addrspace(3) %35, align 4
  br label %37

37:                                               ; preds = %30, %34
  %.128 = phi i32 [ %36, %34 ], [ %.027, %30 ]
  %.1 = phi float [ %32, %34 ], [ %.0, %30 ]
  %38 = add nsw i32 %.026, 1
  br label %28, !llvm.loop !10

39:                                               ; preds = %28
  %40 = fptrunc float %.0 to half
  %41 = getelementptr inbounds half, ptr addrspace(1) %1, i32 %6
  store half %40, ptr addrspace(1) %41, align 2
  %42 = getelementptr inbounds i32, ptr addrspace(1) %2, i32 %6
  store i32 %.027, ptr addrspace(1) %42, align 4
  br label %43

43:                                               ; preds = %39, %3
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z12get_group_idj(i32 noundef) #2

; Function Attrs: convergent nounwind willreturn memory(none)
declare dso_local spir_func i32 @_Z13get_global_idj(i32 noundef) #2

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z16get_sub_group_idv() #3

; Function Attrs: convergent nounwind
declare dso_local spir_func i32 @_Z22get_sub_group_local_idv() #3

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z7barrierj(i32 noundef) #3

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #1 = { alwaysinline convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #2 = { convergent nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #4 = { convergent nounwind willreturn memory(none) "uniform-work-group-size" }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!opencl.ocl.version = !{!0}
!opencl.cxx.version = !{!1}
!opencl.spir.version = !{!0}
!llvm.module.flags = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 3, i32 0}
!1 = !{i32 2021, i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 23.1.1 (https://github.com/llvm/llvm-project 6dfe1677ab8dffbc6ec13d53a1e0215d75147689)"}
!4 = !{i32 1, i32 1, i32 1}
!5 = !{!"none", !"none", !"none"}
!6 = !{!"float16_t*", !"float16_t*", !"int*"}
!7 = !{!"half*", !"half*", !"int*"}
!8 = !{!"const", !"", !""}
!9 = !{!"x", !"tile_values", !"tile_indices"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
