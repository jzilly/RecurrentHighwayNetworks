--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')
local data_format = require('hdf5')
local ptb_path = "./data/"

local vocab_idx = 0
local vocab_map = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size (x_inp:size(1) / batch_size) x batch_size.
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

local function load_data(fname)
 
  local myFile = data_format.open('./data/HutterPrize_Torch.hdf5', 'r')
  local x = myFile:read(fname):all()
  myFile:close()
  return x
end

local function traindataset(batch_size)
   local x = load_data("split/training/default")
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
   local x = load_data("split/test/default")
   --x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
   x = replicate(x, batch_size)
   return x
end

local function validdataset(batch_size)
   local x = load_data("split/validation/default")
   x = replicate(x, batch_size)
   return x
end

return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset}
