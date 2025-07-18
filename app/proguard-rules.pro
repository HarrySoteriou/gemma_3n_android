# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /home/harry/gemma3n_comp_kios/gemma3n/android-app/app/build/intermediates/proguard-files/proguard-android-optimize.txt
# You can edit the file in that directory to keep the flags separated.
# See https://developer.android.com/studio/build/shrink-code.html for more information.

-keep class com.google.common.util.concurrent.ListenableFuture {
    <methods>;
}
