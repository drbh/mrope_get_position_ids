{
 description = "Flake for mrope_get_position_ids kernel";
 inputs = {
   kernel-builder = {
     url = "git+ssh://git@github.com/huggingface/kernel-builder";
     type = "git";
     submodules = true;
   };
 };
 outputs =
   {
     self,
     kernel-builder,
   }:
   kernel-builder.lib.genFlakeOutputs ./.;

 nixConfig = {
   extra-substituters = [ "https://kernel-builder.cachix.org" ];
   extra-trusted-public-keys = [ "kernel-builder.cachix.org-1:JCt71vSCqW9tnmOsUigxf7tVLztjYxQ198FI/j8LrFQ=" ];
 };
}