-- This script implements Recurrent Highway Networks (Zilly and Srivastava et al., 2016) 
-- The main changes to the variational dropout implementation by Gal (2015) (which in turn is based on Zaremba's work) is
-- 1. Using a Recurrent Highway Layer instead of an LSTM
-- 2. Having an initial negative bias for the transfer gates to faciliate learning long-term dependencies
-- 3. Tuning of hyperparameters to adapt to Recurrent Highway Networks
-- All other parts of the code should be identical or close to identical to Gal's implementation.
-- 
-- Single model test BPC for enwik8 is improved from Chung et al.'s 
-- "Hierarchical Multiscale Recurrent Neural Networks" BPC of 1.40 to 1.32. 
-- 
-- References:
-- Zilly, J, Srivastava, R, Koutnik, J, Schmidhuber, J., "Recurrent Highway Networks", 2016
-- Gal, Y, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", 2015.
-- Zaremba, W, Sutskever, I, Vinyals, O, "Recurrent neural network regularization", 2014.

local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
local hutter = require('data_hutter')

local params = {batch_size=100,
                seq_length=50,
                layers=1,
                decay=1.04,
                rnn_size=1500,
                dropout_x=0.1,
                dropout_i=0.3,
                dropout_h=0.05,
                dropout_o=0.3,
                init_weight=0.04,
                lr=0.2,
                vocab_size=205,
                max_epoch=5,
                max_max_epoch=1000,
                max_grad_norm=10,
                weight_decay=0,
                recurrence_depth=10,
		        initial_bias=-4}


-- Yarin: use dropout from within the script rather than nn's
local disable_dropout = false
local function local_Dropout(input, noise)
  return nn.CMulTable()({input, noise})
end

local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx


local function rhn(x, prev_c, prev_h, noise_i, noise_h)
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slice the n_gates dimension, i.e dimension 2
  local reshaped_noise_i = nn.Reshape(2,params.rnn_size)(noise_i)
  local reshaped_noise_h = nn.Reshape(2,params.rnn_size)(noise_h)
  local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i)
  local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)
  -- Calculate all two gates
  local dropped_h_tab = {}
  local h2h_tab = {}
  local t_gate_tab = {}
  local c_gate_tab = {}
  local in_transform_tab = {}
  local s_tab = {}
  for layer_i = 1, params.recurrence_depth do
    local i2h        = {}
    h2h_tab[layer_i] = {}
    if layer_i == 1 then
      for i = 1, 2 do
        -- Use select table to fetch each gate
        local dropped_x         = local_Dropout(x, nn.SelectTable(i)(sliced_noise_i))
        dropped_h_tab[layer_i]  = local_Dropout(prev_h, nn.SelectTable(i)(sliced_noise_h))
        i2h[i]                  = nn.Linear(params.rnn_size, params.rnn_size)(dropped_x)
        h2h_tab[layer_i][i]     = nn.Linear(params.rnn_size, params.rnn_size)(dropped_h_tab[layer_i])
      end
      t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(nn.CAddTable()({i2h[1], h2h_tab[layer_i][1]})))
      in_transform_tab[layer_i] = nn.Tanh()(nn.CAddTable()({i2h[2], h2h_tab[layer_i][2]}))
      c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i]))
      s_tab[layer_i]           = nn.CAddTable()({
        nn.CMulTable()({c_gate_tab[layer_i], prev_h}),
        nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
      })
    else
      for i = 1, 2 do
        -- Use select table to fetch each gate
        dropped_h_tab[layer_i]  = local_Dropout(s_tab[layer_i-1], nn.SelectTable(i)(sliced_noise_h))
        h2h_tab[layer_i][i]     = nn.Linear(params.rnn_size, params.rnn_size)(dropped_h_tab[layer_i])
      end
      t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(h2h_tab[layer_i][1]))
      in_transform_tab[layer_i] = nn.Tanh()(h2h_tab[layer_i][2])
      c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i]))
      s_tab[layer_i]           = nn.CAddTable()({
        nn.CMulTable()({c_gate_tab[layer_i], s_tab[layer_i-1]}),
        nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
      })
    end
  end
  local next_h = s_tab[params.recurrence_depth]
  local next_c = prev_c
  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local noise_x          = nn.Identity()()
  local noise_i          = nn.Identity()()
  local noise_h          = nn.Identity()()
  local noise_o          = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                              params.rnn_size)(x)}
  i[0] = local_Dropout(i[0], noise_x)
  local next_s           = {}
  local split            = {prev_s:split(2 * params.layers)}
  local noise_i_split    = {noise_i:split(params.layers)}
  local noise_h_split    = {noise_h:split(params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local n_i            = noise_i_split[layer_idx]
    local n_h            = noise_h_split[layer_idx]
    local next_c, next_h = rhn(i[layer_idx - 1], prev_c, prev_h, n_i, n_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = local_Dropout(i[params.layers], noise_o)
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s, noise_x, noise_i, noise_h, noise_o},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

local function setup()
  print("Creating an RHN network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.noise_i = {}
  model.noise_x = {}
  model.noise_xe = {}
  for j = 1, params.seq_length do
    model.noise_x[j] = transfer_data(torch.zeros(params.batch_size, 1))
    model.noise_xe[j] = torch.expand(model.noise_x[j], params.batch_size, params.rnn_size)
    model.noise_xe[j] = transfer_data(model.noise_xe[j])
  end
  model.noise_h = {}
  for d = 1, params.layers do
    model.noise_i[d] = transfer_data(torch.zeros(params.batch_size, 2 * params.rnn_size))
    model.noise_h[d] = transfer_data(torch.zeros(params.batch_size, 2 * params.rnn_size))
  end
  model.noise_o = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))

  model.pred = {}
  for j = 1, params.seq_length do
    model.pred[j] = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
  end
  local y                = nn.Identity()()
  local pred             = nn.Identity()()
  local err              = nn.ClassNLLCriterion()({pred, y})
  model.test             = transfer_data(nn.gModule({y, pred}, {err}))
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

-- convenience functions to handle noise
local function sample_noise(state)
  for i = 1, params.seq_length do
    model.noise_x[i]:bernoulli(1 - params.dropout_x)
    model.noise_x[i]:div(1 - params.dropout_x)
  end

  for b = 1, params.batch_size do
    for i = 1, params.seq_length do
      local x = state.data[state.pos + i - 1]
      for j = i+1, params.seq_length do
        if state.data[state.pos + j - 1] == x then
          model.noise_x[j][b] = model.noise_x[i][b]
          -- we only need to override the first time; afterwards subsequent are copied:
          break
        end
      end
    end
  end
  for d = 1, params.layers do
    model.noise_i[d]:bernoulli(1 - params.dropout_i)
    model.noise_i[d]:div(1 - params.dropout_i)
    model.noise_h[d]:bernoulli(1 - params.dropout_h)
    model.noise_h[d]:div(1 - params.dropout_h)
  end
  model.noise_o:bernoulli(1 - params.dropout_o)
  model.noise_o:div(1 - params.dropout_o)
end

local function reset_noise()
  for j = 1, params.seq_length do
    model.noise_x[j]:zero():add(1)
  end
  for d = 1, params.layers do
    model.noise_i[d]:zero():add(1)
    model.noise_h[d]:zero():add(1)
  end
  model.noise_o:zero():add(1)
end

local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  if disable_dropout then reset_noise() else sample_noise(state) end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward(
      {x, y, s, model.noise_xe[i], model.noise_i, model.noise_h, model.noise_o}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err
end

local function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward(
      {x, y, s, model.noise_xe[i], model.noise_i, model.noise_h, model.noise_o},
      {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
  paramx:add(-params.weight_decay, paramx)
end

local function run_valid()
  reset_state(state_valid)
  disable_dropout = true
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    local p = fp(state_valid)
    perp = perp + p:mean()
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  print("Validation set loss (BPC) : " .. g_f3((perp / len)/torch.log(2)))
  disable_dropout = false
end

local function run_test()
  reset_state(state_test)
  disable_dropout = true
  local len = (state_test.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    local p = fp(state_test)
    perp = perp + p:mean()
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / len)))
  print("Test set loss (BPC) : " .. g_f3((perp / len)/torch.log(2)))
  disable_dropout = false

end

local function main()
  g_init_gpu(1)
  state_train = {data=transfer_data(hutter.traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(hutter.validdataset(params.batch_size))}
  state_test =  {data=transfer_data(hutter.testdataset(params.batch_size))}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_train):mean()
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
  	    ', Training set loss (BPC) = ' .. g_f3(perps:mean()/torch.log(2)) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      run_test()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  run_test()
  print("Training is over.")
end

main()
